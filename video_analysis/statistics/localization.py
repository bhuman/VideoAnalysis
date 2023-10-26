from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cairo
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from video_analysis.world_model.players import Player

if TYPE_CHECKING:
    import numpy.typing as npt

    from video_analysis.camera import Camera
    from video_analysis.world_model import WorldModel


class Localization:
    """This class implements a provider for the average localization error per team."""

    class PlayerStatus:
        """The status of a player that is mostly based on GameController data.

        Objects of this class are updated whenever the GameController log plays back
        a return packet, i.e. a packet that contains the position on the field the
        player believes to be located at.
        """

        def __init__(self, position: npt.NDArray[np.float_], distances: dict[Player, float]) -> None:
            """Constructor of a player status object.

            :param position: The position of this player reported to the GameController.
            :param distances: The distances to all world model players.
            """
            self.position: npt.NDArray[np.float_] = position
            self.distances: dict[Player, float] = distances
            self.penalized: bool = True  # Is the player penalized?
            self.returning_soon: bool = False  # Will the player return soon from its penalty?
            self.color: Player.Color = Player.Color.UNKNOWN  # The color of this player.
            self.player: Player | None = None  # The associated world model player.
            self.player_since: float = 0  # Since when is the same player associated?
            self.player_valid: bool = False  # Association to player is valid

        def draw_on_field(self, context: cairo.Context, player_num: str) -> None:
            """Draw this player status onto the field.

            :param context: The context to draw to.
            :param player_num: The player number as string.
            """
            x = self.position[0]
            y = self.position[1]

            context.save()
            context.set_line_width(0.02)
            context.set_source_rgb(*self.color.as_color_triple(range01=True))
            context.arc(x, y, 0.15, 0, 2 * np.pi)
            context.stroke()
            context.move_to(x - 55 / 1000, y - 65 / 1000)
            context.show_text(player_num)
            context.stroke()
            if self.player_valid and self.player is not None:
                context.move_to(self.player.position[0], self.player.position[1])
                context.line_to(x, y)
                context.set_source_rgb(*self.color.as_color_triple(range01=True))
                context.set_line_width(0.04)
                context.stroke()
            context.restore()

        def draw_on_image(self, image: npt.NDArray[np.uint8], camera: Camera) -> None:
            """Draw this player status onto the image.

            :param image: The image to draw onto.
            :param camera: Used to transform field coordinates into image coordinates.
            """
            radius = 0.17
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            points_in_world = self.position + np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius
            points_in_image = camera.world2image(points_in_world).astype(np.int32)
            color = self.color.as_color_triple(bgr=True)
            for p1, p2 in zip(points_in_image[0::2], points_in_image[1::2]):
                cv2.line(image, pt1=p1, pt2=p2, color=color, thickness=3)
            if self.player_valid and self.player is not None:
                centers_in_world = np.stack([self.position, self.player.position], axis=0)
                centers_in_image = camera.world2image(centers_in_world).astype(np.int32)
                cv2.line(image, centers_in_image[0], centers_in_image[1], color=color, thickness=3)

    _player_valid_delay = 2
    """Time in seconds a world model player has to associated to a status to count."""

    _returning_thresholds: list[float] = [-0.1, 0.5]
    """Thresholds around points where robots return from penalties ((x, y) in m)."""

    _penalized_thresholds: list[float] = [1, 0.2]
    """Thresholds around points where robots stand when penalized ((x, y) in m)."""

    def __init__(self, world_model: WorldModel, categories: dict[str, list], settings: dict[str, Any]) -> None:
        """Constructor of the localization statistics.

        :param world_model The world model that contains all the seen players.
        :param categories: The statistics category provided by this provider is added to the
        dictionary of all categories.
        :param settings: The settings are used to access a constant of the log parser.
        """
        self._world_model: WorldModel = world_model
        self._categories: dict[str, list] = categories
        self._categories["Mean Localization Error"] = [0, 0, "m", "Mean Loc. Error"]
        self._settings: dict[str, Any] = settings
        self._player_statuses: list[dict[str, Localization.PlayerStatus]] = [{}, {}]
        self._overall_cost: list[float] = [0.0, 0.0]  # Sum of all localization errors per team.
        self._n_cost_samples: list[int] = [0, 0]  # The number of all localization errors per team.
        self._timeout_budget: int = -1  # The sum of all remaining timeouts.
        self._game_since: float = -10000  # When started the current part of the game?

    def update(self) -> None:
        """Updates the statistics."""
        if self._world_model.game_state.changed:
            # Detect whether there was a timeout.
            current_budget = (
                self._world_model.game_state.current_game_state["teams"]["home"]["timeoutBudget"]
                + self._world_model.game_state.current_game_state["teams"]["away"]["timeoutBudget"]
            )
            if self._timeout_budget != current_budget:
                # If this is not the start of the half, reinitialize statuses.
                if self._timeout_budget != -1:
                    self._game_since = self._world_model.timestamp
                    self._player_statuses = [{}, {}]
                self._timeout_budget = current_budget

        # Make sure not to mix position data recorded after a timeout with video data recorded before it.
        if self._world_model.timestamp - self._game_since >= self._settings["log_parser"]["video_stop_delay"]:
            for index, team in enumerate(self._player_statuses):
                for player in team:
                    self._update_status(index, int(player))
            if self._world_model.game_state.current_status_messages:
                self._handle_status_messages(status_messages=self._world_model.game_state.current_status_messages)

    def _handle_status_messages(self, status_messages: list[dict[str, Any]]) -> None:
        """Process all new messages that players sent to the GameController.

        Should only be called if there are actually new messages.
        For each message, the status of the corresponding player is updated, or if it does not exist,
        a new one is created.
        :param status_messages: The new messages replayed from the log file.
        """
        # Filter out world model players that might result from penalized robots.
        players: list[Player] = self.filter_players(self._world_model.players.players)

        # Update the player statuses.
        for status_message in status_messages:
            data: tuple[str, int, int, int, bool, float, float, float, float, float, float] = status_message["data"]
            player_num: int = data[2]
            team_num: int = data[3]
            index: int = 0 if self._world_model.game_state.teams[0].number == team_num else 1
            side: int = 1 if index == 0 else -1
            position: npt.NDArray[np.float_] = np.array([data[5] * side / 1000, data[6] * side / 1000])
            distances: dict[Player, float] = {
                player: np.linalg.norm(position - player.position).astype(float) for player in players
            }
            if str(player_num) not in self._player_statuses[index]:
                self._player_statuses[index][str(player_num)] = Localization.PlayerStatus(position, distances)
                if self._world_model.game_state.current_game_state:
                    self._update_status(index, player_num)
            else:
                status: Localization.PlayerStatus = self._player_statuses[index][str(player_num)]
                status.position = position
                status.distances = distances

        # Solve the assignment problem.
        self._solve()

    def filter_players(self, players: list[Player]) -> list[Player]:
        """Filter out world model players that might result from penalized robots.

        It is first determined for each team, whether there are actually penalized players.
        It is also determined, whether some of them might return soon to the game. Such
        players could already be placed on the sideline, while others are expected to be
        standing further away from the field.
        """
        penalized: list[bool] = [False, False]
        returning_soon: list[bool] = [False, False]
        for index, team in enumerate(self._player_statuses):
            penalized[index] = any(map(lambda status: team[status].penalized, team))
            returning_soon[index] = any(map(lambda status: team[status].returning_soon, team))

        # The x coordinate of the penalty cross is used as center of region in which players are ignored.
        cross_x: float = self._world_model.field.field_length / 2 - self._world_model.field.penalty_cross_distance

        # The dimension of that region depends on the y coordinate of the sideline.
        sideline_y: float = self._world_model.field.field_width / 2

        filtered_players: list[Player] = []
        for player in players:
            for index, team in enumerate(self._player_statuses):
                # For now, ignore the actual player colors, because they might be misclassified when a robot
                # stands next to the field.

                # The position of the penalty cross depends on the team with a penalty.
                cross_x = abs(cross_x) * (1 if index == 1 else -1)

                # Check the two types of regions, each of which is on both (y) sides of the field.
                if (
                    returning_soon[index]
                    and abs(player.position[0] - cross_x) <= self._returning_thresholds[0]
                    and abs(player.position[1]) >= sideline_y + self._returning_thresholds[1]
                ):
                    break  # Exclude
                if (
                    penalized[index]
                    and abs(player.position[0] - cross_x) <= self._penalized_thresholds[0]
                    and abs(player.position[1]) >= sideline_y + self._penalized_thresholds[1]
                ):
                    break  # Exclude
            else:
                # No reason the filter out the player found: keep it.
                filtered_players.append(player)

        return filtered_players

    def _update_status(self, team: int, player_num: int) -> None:
        """Update a player status based on GameController information.

        Updates the penalized status and the jersey color.
        :param team: The side index of the team (0 = left, 1 = right).
        :param player_num: The number of the player (1-based).
        """
        side: str = (
            "home"
            if (team == 0) == (self._world_model.game_state.current_game_state["sides"] == "homeDefendsLeftGoal")
            else "away"
        )
        player: dict[str, Any] = self._world_model.game_state.current_game_state["teams"][side]["players"][
            player_num - 1
        ]
        penalized: bool = player["penalty"] != "noPenalty" and player["penalty"] != "motionInSet"
        returnUncertain: bool = player["penalty"] == "substitute" or player["penalty"] == "pickedUp"
        remaining: int = (
            0 if player["penaltyTimer"] == "stopped" else player["penaltyTimer"]["!started"]["remaining"][0]
        )
        self._player_statuses[team][str(player_num)].penalized = penalized
        self._player_statuses[team][str(player_num)].returning_soon = (
            penalized and not returnUncertain and remaining < 10
        )
        goalkeeper: bool = self._world_model.game_state.current_game_state["teams"][side]["goalkeeper"] == player_num
        self._player_statuses[team][str(player_num)].color = Player.Color.from_str(
            self._world_model.game_state.teams[team].goalkeeper_color
            if goalkeeper
            else self._world_model.game_state.teams[team].field_player_color
        )

    def _solve(self) -> None:
        """Solve the assignment problem.

        Finds an assignment with minimal error between the positions reported by the players
        and their world model positions that were detected in the video. Since positions are
        not all reported at the same time, each reported position has its own state of the
        world. When a status was played back, distances to all world model players were
        computed and stored in the status. A different points in time, a different set of
        world model players can be present. Therefore, it is first determined, which were
        present in all the statuses. Only those are assigned to avoid mismatches.
        Penalized players are not assigned.
        """
        # Compute the set of players present in all statuses.
        always_seen: set[Player] | None = None
        for team in self._player_statuses:
            for status in team:
                if not team[status].penalized:
                    once_seen = [seen for seen in team[status].distances]
                    if always_seen is None:
                        always_seen = set(once_seen)
                    else:
                        always_seen = always_seen.intersection(set(once_seen))
        seen_players: list[Player] = list(always_seen) if always_seen is not None else []

        # Create the cost matrix, i.e. the matrix with distances between status positions
        # and world model positions. Rows represent statuses. Columns represent world model
        # players.
        cost_matrix = np.array(
            [
                [team[status].distances[seen] for seen in seen_players]
                for team in self._player_statuses
                for status in team
                if not team[status].penalized
            ]
        )

        # Compute the best match. The result are pairs of (row, column).
        status_player_matches: list[tuple[int, int]] = list(zip(*linear_sum_assignment(cost_matrix)))

        # Filter the assignment and update the costs.
        # Assignments are only used if they have not changed for a while, the
        # player is upright, and the status and the world model player belong
        # to the same team (color-wise).
        row: int = 0
        for index, team in enumerate(self._player_statuses):
            team_colors: list[Player.Color] = [
                Player.Color.from_str(self._world_model.game_state.teams[index].goalkeeper_color),
                Player.Color.from_str(self._world_model.game_state.teams[index].field_player_color),
            ]
            for status in team:
                if team[status].penalized:
                    team[status].player = None
                    team[status].player_valid = False
                else:
                    for s, p in status_player_matches:
                        if s == row:
                            if team[status].player != seen_players[p]:
                                team[status].player_since = self._world_model.timestamp
                            if seen_players[p].upright and seen_players[p].color in team_colors:
                                team[status].player = seen_players[p]
                                team[status].player_valid = (
                                    self._world_model.timestamp - team[status].player_since >= self._player_valid_delay
                                )
                                if team[status].player_valid:
                                    self._overall_cost[index] += cost_matrix[s][p]
                                    self._n_cost_samples[index] += 1
                            else:
                                team[status].player_valid = False
                            break
                    else:
                        team[status].player_since = self._world_model.timestamp
                        team[status].player_valid = False
                    row += 1
            if self._n_cost_samples[index] > 0:
                self._categories["Mean Localization Error"][index] = round(
                    self._overall_cost[index] / self._n_cost_samples[index], 2
                )

    def draw_on_field(self, context: cairo.Context) -> None:
        """Draw the player statuses onto the field.

        :param context: The context to draw to.
        """
        for team in self._player_statuses:
            for status in team:
                if not team[status].penalized:
                    team[status].draw_on_field(context=context, player_num=status)

    def draw_on_image(self, image: npt.NDArray[np.uint8]) -> None:
        """Draw the player statuses onto the image.

        :param image: The image to draw onto.
        """
        for team in self._player_statuses:
            for status in team:
                if not team[status].penalized:
                    team[status].draw_on_image(image=image, camera=self._world_model.camera)
