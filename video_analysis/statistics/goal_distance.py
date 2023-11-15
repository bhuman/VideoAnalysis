from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..world_model import WorldModel


class GoalDistance:
    """This class implements a provider for the average distance between ball and own goal per team."""

    def __init__(self, world_model: WorldModel, categories: dict[str, list]) -> None:
        """Initialize the provider for the average distance between ball and own goal per team.

        :param world_model: The world model.
        :param categories: The statistics category provided by this object is added to this dictionary
        of all categories.
        """
        self._world_model: WorldModel = world_model
        self._categories: dict[str, list] = categories
        self._categories.update({"Distance to Own Goal": [0, 0, " m", "Dist. to Own Goal"]})
        self._categories.update({"Ball Distance to Own Goal": [0, 0, " m", "  Ball"]})
        self._total_ball_to_own_goal = [0.0, 0.0]  # first team, second team
        self._total_players_to_own_goal = [0.0, 0.0]  # first team, second team
        self._ball_counter: int = 0
        self._players_counter: list[int] = [0, 0]  # first team, second team
        gp_x = self._world_model.field.field_length * 0.5
        gp_y = self._world_model.field.goal_inner_width * 0.5
        self._goalposts = [
            [np.array([-gp_x, -gp_y]), np.array([-gp_x, gp_y])],
            [np.array([gp_x, -gp_y]), np.array([gp_x, gp_y])],
        ]

    def update(self) -> None:
        """Update the average own goal distance category."""

        # Skip if not in playing state.
        if self._world_model.game_state.state != self._world_model.game_state.State.PLAYING:
            return

        # Compute sum of distances between ball and goals if the ball is seen.
        if self._world_model.ball.last_seen == self._world_model.timestamp:
            for team in range(2):
                self._total_ball_to_own_goal[team] += self._distance_to_goal(team, self._world_model.ball.position)
            self._ball_counter += 1

        # Compute sum of distances between players and own goals.
        for player in self._world_model.players:
            if abs(player.position[1]) < self._world_model.field.field_width / 2:
                team = self._world_model.players.color_to_team(player.color)
                self._total_players_to_own_goal[team] += self._distance_to_goal(team, player.position)
                self._players_counter[team] += 1

        # Update category values.
        for team in range(2):
            self._categories["Ball Distance to Own Goal"][team] = round(
                self._total_ball_to_own_goal[team] / max(1, self._ball_counter),
                2,
            )
            self._categories["Distance to Own Goal"][team] = round(
                self._total_players_to_own_goal[team] / max(1, self._players_counter[team]),
                2,
            )

    def _distance_to_goal(self, side: int, position: npt.NDArray[np.float_]) -> float:
        """Calculate the distance between a position and a goal.

        :param side: The side of the goal (0 = left, 1 = right).
        :param position: The position the distance is calculated to.
        :return: The distance between the position and the goal.
        """
        if abs(position[1]) > abs(self._goalposts[side][0][1]):
            return np.linalg.norm(self._goalposts[side][position[1] > 0] - position).astype(float)
        else:
            return abs(self._goalposts[side][0][0] - position[0])
