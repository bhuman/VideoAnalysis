from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..world_model import WorldModel
    from .possession import Possession


class DistanceCounter:
    """This class determines the distance that the ball and the players of each team have moved.

    This is only done when the game is in the playing state. The ball movement is also associated
    with the teams based on the assumed ball possession. The movement without ball possession is
    also tracked, but not used.
    """

    _min_distance = 0.1
    """The minimum distance that is considered "moved" to filter out noise."""

    def __init__(self, world_model: WorldModel, categories: dict[str, list]) -> None:
        """Initialize the provider to track ball possession.

        :param world_model: The world model.
        :param categories: The statistics categories provided by this object are added to this
        dictionary of all categories. However, this is done in `set_possession`, to keep the
        desired order for the UI.
        """
        self._world_model = world_model
        self._categories: dict[str, list] = categories
        self._possession: Possession | None = None
        self._ball: npt.NDArray[np.float_] | None = None
        self._players: dict[int, npt.NDArray[np.float_]] = {}
        """The last position per player."""

        self._ball_distances: list[float] = [0, 0, 0]
        """Meters the ball was moved by first, second, unknown team."""

        self._player_distances: list[float] = [0, 0]
        """Meters all players of each team traveled."""

        self.time_ball_stopped: float = 0
        """Number of seconds the ball was not moving."""

    def set_possession(self, possession: Possession) -> None:
        """Remember the reference to the provider of the ball possession.

        Also registers the statistics categories provided by this object.
        :param possession: The provider of the ball possession statistics.
        """
        self._possession = possession
        self._categories.update(
            {
                "Distance Ball Moved": [0, 0, " m", "  Distance Moved"],
                "Distance Walked": [0, 0, " m"],
            }
        )

    def update(self) -> None:
        """Update the distance categories."""
        if self._world_model.game_state.state == self._world_model.game_state.State.PLAYING:
            if self._world_model.ball.last_seen == self._world_model.timestamp:
                # If this is the first time this method updates the ball,
                # make sure the ball position is a realistic value:
                if self._ball is None:
                    self._ball = self._world_model.ball.position

                # Compute distance between last saved position and current position.
                distance_ball: float = np.linalg.norm(self._world_model.ball.position - self._ball).astype(float)

                # Ignore noise in the detected movements.
                if distance_ball > self._min_distance:
                    self._ball = self._world_model.ball.position
                    assert self._possession is not None
                    self._ball_distances[self._possession.state] += distance_ball
                    self.time_ball_stopped = 0
                else:
                    self.time_ball_stopped += 1 / self._world_model.camera.fps

            for player in self._world_model.players:
                if not player.upright:
                    self._players.pop(player.id_, None)
                else:
                    if player.upright and player.id_ not in self._players:
                        self._players[player.id_] = player.position

                    # Compute distance between last saved position and current position.
                    distance_player: float = np.linalg.norm(player.position - self._players[player.id_]).astype(float)

                    # Ignore noise in the detected movements.
                    if distance_player >= self._min_distance:
                        self._players[player.id_] = player.position
                        self._player_distances[self._world_model.players.color_to_team(player.color)] += distance_player
        else:
            # Forget everything when not in playing state.
            self._ball = None
            self._players = {}

        # Update statistics categories.
        for team in range(2):
            self._categories["Distance Ball Moved"][team] = round(self._ball_distances[team], 1)
            self._categories["Distance Walked"][team] = round(self._player_distances[team], 1)
