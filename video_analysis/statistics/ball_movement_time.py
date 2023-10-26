from ..world_model import WorldModel
from .distance_counter import DistanceCounter
from .possession import Possession


class BallMovementTime:
    def __init__(
        self,
        world_model: WorldModel,
        distance_counter: DistanceCounter,
        categories: dict[str, list],
    ) -> None:
        self._world_model = world_model
        self._distance_counter = distance_counter
        self._categories: dict[str, list] = categories

        self._ball_moved_times: list[float] = [0, 0, 0]

    def set_possession(self, possession: Possession) -> None:
        """Remember the reference to the provider of the ball possession.

        Also registers the statistics categories provided by this object.
        :param possession: The provider of the ball possession statistics.
        """
        self._possession = possession
        self._categories.update({"Time Ball Moved": [0, 0, " s", "  Time Moved"]})

    def update(self) -> None:
        if self._world_model.game_state.state == self._world_model.game_state.State.PLAYING:
            if self._world_model.ball.last_seen == self._world_model.timestamp:

                # If the ball has not moved, the time would not be zero
                has_ball_moved = self._distance_counter.time_ball_stopped == 0

                # Using the possession state, the time for the corresponding team is updated
                if has_ball_moved:
                    self._ball_moved_times[self._possession.state] += 1 / self._world_model.camera.fps

        for team in range(2):
            self._categories["Time Ball Moved"][team] = round(self._ball_moved_times[team])
