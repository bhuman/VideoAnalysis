from __future__ import annotations

import numpy as np

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
        self._total_distance_to_own_goal = [0.0, 0.0]  # first team, second team
        self._time: float = 0

    def update(self) -> None:
        """Update the average own goal distance category."""

        # Skip if not in playing state of the ball is not seen.
        if (
            self._world_model.game_state.state != self._world_model.game_state.State.PLAYING
            or self._world_model.ball.last_seen != self._world_model.timestamp
        ):
            return

        # Compute sum of distances, weighted by time (technically meter seconds).
        gp_x = self._world_model.field.field_length * 0.5
        gp_y = self._world_model.field.goal_inner_width * 0.5
        gps = [[np.array([-gp_x, -gp_y]), np.array([-gp_x, gp_y])], [np.array([gp_x, -gp_y]), np.array([gp_x, gp_y])]]
        self._time += 1 / self._world_model.camera.fps
        for i, gp in enumerate(gps):
            if abs(self._world_model.ball.position[1]) > gp_y:
                self._total_distance_to_own_goal[i] += (
                    np.linalg.norm(gp[self._world_model.ball.position[1] > 0] - self._world_model.ball.position).astype(
                        float
                    )
                    / self._world_model.camera.fps
                )
            else:
                self._total_distance_to_own_goal[i] += (
                    abs(gp[0][0] - self._world_model.ball.position[0]) / self._world_model.camera.fps
                )

        # Update category values.
        for team in range(2):
            self._categories["Distance to Own Goal"][team] = round(
                self._total_distance_to_own_goal[team] / max(1, self._time),
                2,
            )
