from __future__ import annotations

from typing import TYPE_CHECKING

import cairo
import cv2
import numpy as np
import numpy.typing as npt

from ..world_model import WorldModel
from ..world_model.players import Player

if TYPE_CHECKING:
    from .ball_movement_time import BallMovementTime
    from .distance_counter import DistanceCounter


class Possession:
    """This class creates statistics about ball possession."""

    NONE = 2
    """The state if no team is in possession of the ball."""

    def __init__(
        self,
        world_model: WorldModel,
        distance_counter: DistanceCounter,
        ball_movement_time: BallMovementTime,
        categories: dict[str, list],
    ) -> None:
        """Initialize the provider to track ball possession.

        :param world_model: The world model.
        :param distance_counter: An object that determines whether the ball is moving. There is a cyclic dependency
        with that object, because it also uses information about ball possession. Therefore, both have a reference
        to the other.
        :param ball_movement_time: An object that determines how long the ball is moving. To register the categories
        in the desired order, this object calls the method that will register that category.
        :param categories: The statistics categories that are extended by this object. This object will also add
        the categories for `distance_counter` to make them appear after the category of this object.
        """
        self._world_model: WorldModel = world_model
        self._categories: dict[str, list] = categories
        self._categories.update({"Passes": [0, 0, ""]})
        self._categories.update({"Ball Possession": [0, 0, " s"]})
        self._distance_counter: DistanceCounter = distance_counter

        # This call will register `ball_movement_time`'s category.
        ball_movement_time.set_possession(self)

        # This call will register `distance_counter`'s categories.
        distance_counter.set_possession(self)

        self.state: int = self.NONE
        self._possession_times: list[float] = [0, 0, 0]  # first team, second team, NONE
        self._passes: list[int] = [0, 0, 0]
        self._timestamp: float = 0
        self._ball_stopped: bool = True
        self._time_when_ball_stopped: float = 0
        self._player_in_ball_possession: Player | None = None
        self._draw_player_in_ball_possession: bool = False
        self._kick_position: npt.NDArray[np.float_] = self._world_model.ball.position
        self._kick_distance: float = 0

    def update(self) -> None:
        """Update the ball possession category."""
        if self._world_model.game_state.state == self._world_model.game_state.State.PLAYING:
            self._timestamp += 1 / self._world_model.camera.fps

            # Determine whether ball is moving or not.
            assert self._distance_counter
            if self._distance_counter.time_ball_stopped >= 0.2:
                if not self._ball_stopped:
                    self._time_when_ball_stopped = self._timestamp - self._distance_counter.time_ball_stopped
                    self._ball_stopped = True
                    self._kick_distance = np.linalg.norm(self._world_model.ball.position - self._kick_position).astype(
                        float
                    )
                    self._draw_player_in_ball_possession = False
            else:
                self._ball_stopped = False
                self._time_when_ball_stopped = self._timestamp

            # Update current possession state.
            self._update_state()

            if self._ball_stopped and self._kick_distance == 0:
                self._kick_position = self._world_model.ball.position

            # Update statistics category.
            for team in range(2):
                self._categories["Ball Possession"][team] = round(self._possession_times[team])
                self._categories["Passes"][team] = self._passes[team]
        else:
            self.state = self.NONE
            self._player_in_ball_possession = None
            self._draw_player_in_ball_possession = False

    def _update_state(self) -> None:
        """Update the ball possession state."""
        time_since_ball_stopped = self._timestamp - self._time_when_ball_stopped

        # Find closest players and distances for both teams.
        closest_players: list[Player | None] = [None, None]
        distances: list[float] = [0, 0]
        for team in range(2):
            closest_players[team], distances[team] = self._get_closest(
                [
                    player
                    for player in self._world_model.players
                    if self._world_model.players.color_to_team(player.color) == team and player.upright
                ],
                self._world_model.ball.position,
            )

        # State machine for ball possession.
        if self.state == self.NONE:
            if min(distances) < 0.4 and self._ball_stopped:
                self.state = 0 if distances[0] < distances[1] else 1
        elif distances[self.state] > 0.6 and time_since_ball_stopped > 0.6 and distances[1 - self.state] < 0.4:
            self.state = 1 - self.state
            self._player_in_ball_possession = None
            self._draw_player_in_ball_possession = False
            self._kick_distance = 0
        elif distances[self.state] > 0.6 and time_since_ball_stopped > 1:
            self.state = self.NONE
            self._kick_distance = 0

        # Update closest player for drawing methods.
        if self.state < 2 and time_since_ball_stopped > 0.6 and distances[self.state] < 0.4:
            if (
                self._kick_distance > 0.75
                and self._player_in_ball_possession is not None
                and self._player_in_ball_possession != closest_players[self.state]
            ):
                self._passes[self.state] += 1
            self._kick_distance = 0
            self._player_in_ball_possession = closest_players[self.state]
            self._draw_player_in_ball_possession = True
        elif self.state == self.NONE:
            self._draw_player_in_ball_possession = False

        # Update one of the possession times.
        self._possession_times[self.state] += 1 / self._world_model.camera.fps

    def _get_closest(self, players: list[Player], ball: npt.NDArray[np.float_]) -> tuple[Player | None, float]:
        """Find the closest player and its distance to the ball.

        :param players: The list of players to search for the closest one to the ball.
        :param ball: The position of the ball.
        :return: The closest player and the distance of that player from the ball. The closest
        player is `None` if none could be found.
        """
        closest_player = None
        closest_distance: float = 10000.0
        if ball is not None:
            for player in players:
                distance_to_ball: float = np.linalg.norm(player.position - ball).astype(float)
                if distance_to_ball < closest_distance:
                    closest_player = player
                    closest_distance = distance_to_ball
        return closest_player, closest_distance

    def draw_on_field(self, context: cairo.Context) -> None:
        """Draw a line between the ball and the closest player onto the field.

        :param context: The context to draw to.
        """
        if (
            self._draw_player_in_ball_possession
            and self._player_in_ball_possession is not None
            and self._world_model.ball.last_seen == self._world_model.timestamp
        ):
            context.save()
            context.move_to(self._player_in_ball_possession.position[0], self._player_in_ball_possession.position[1])
            context.line_to(self._world_model.ball.position[0], self._world_model.ball.position[1])
            context.set_source_rgb(1, 0.5, 0)
            context.set_line_width(0.04)
            context.stroke()
            context.restore()

    def draw_on_image(self, image: npt.NDArray[np.uint8]) -> None:
        """Draw a line between the ball and the closest player onto the image.

        :param image: The image to draw to.
        """
        if (
            self._draw_player_in_ball_possession
            and self._player_in_ball_possession is not None
            and self._player_in_ball_possession in self._world_model.players
            and self._world_model.ball.last_seen == self._world_model.timestamp
        ):
            player_in_image = self._world_model.camera.world2image(self._player_in_ball_possession.position)
            ball_in_image = self._world_model.camera.world2image(self._world_model.ball.position)
            cv2.line(image, player_in_image.astype(np.int32), ball_in_image.astype(np.int32), (0, 128, 255), 3)
