from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..world_model import WorldModel


class BallIsolationTime:
    """This class counts for how many seconds no player has been closer than 1 m to the ball (total).

    This statistics is special, because it is not directly linked to one of the teams. Therefore, it
    will not be shown in the user interface, but still added to output file (for both teams).
    """

    def __init__(self, world_model: WorldModel, categories: dict[str, list]) -> None:
        """Initialize the provider for the ball isolation time.

        :param world_model: The world model.
        :param categories: The statistics category provided by this provider is added to this dictionary
        of all categories.
        """
        self._world_model: WorldModel = world_model
        self._categories: dict[str, list] = categories
        self._categories.update({"Ball Isolation Time": [0]})
        self._isolated_time: float = 0
        self._isolated_time_consecutive: float = 0

    def update(self) -> None:
        """Update the ball isolation time category."""
        isolated: bool = False
        if self._world_model.ball.last_seen == self._world_model.timestamp:
            for player in self._world_model.players:  # Iterate through players
                distance = np.linalg.norm(player.position - self._world_model.ball.position)
                if distance < 1:  # Ball is considered isolated if no player is closer than 1 meter
                    break
            else:
                isolated = True
        if isolated:
            self._isolated_time_consecutive += 1 / self._world_model.camera.fps
            if self._isolated_time_consecutive >= 0.2:
                self._isolated_time += 1 / self._world_model.camera.fps
        else:
            self._isolated_time_consecutive = 0
        self._categories["Ball Isolation Time"][0] = round(self._isolated_time, 0)
