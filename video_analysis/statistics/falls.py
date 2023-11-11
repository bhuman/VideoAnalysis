from __future__ import annotations

from dataclasses import dataclass

from ..world_model import WorldModel


class Falls:
    """Count the number of falls per team."""

    @dataclass
    class Status:
        """Fall-related status information per player."""

        upright: bool = True
        last_upright: float = 0
        last_fallen: float = 0

    _state_change_delay: int = 1  # 1 second
    """Number of seconds until a state change is accepted."""

    def __init__(self, world_model: WorldModel, categories: dict[str, list]) -> None:
        """Initialize the provider to count falls per team.

        :param world_model: The world model containing upright/fallen per player.
        :param categories: The statistics category provided by this provider is added to the
        dictionary of all categories.
        """
        self._world_model: WorldModel = world_model
        self._categories: dict[str, list] = categories
        self._categories.update({"Falls": [0, 0]})
        self._player: dict[int, Falls.Status] = {}

    def update(self) -> None:
        """Update the fall counters."""
        for player in self._world_model.players.players:
            if player.id_ != -1:
                # Add player if not present.
                if self._player.get(player.id_) is None:
                    self._player[player.id_] = Falls.Status()

                status = self._player[player.id_]

                # Update player status.
                if player.upright:
                    status.last_upright = self._world_model.timestamp
                else:
                    status.last_fallen = self._world_model.timestamp

                # Update upright state and count falls.
                if (
                    status.upright
                    and status.last_upright != 0
                    and self._world_model.timestamp - status.last_upright >= self._state_change_delay
                ):
                    status.upright = False
                    self._categories["Falls"][self._world_model.players.color_to_team(player.color)] += 1
                elif (
                    not status.upright
                    and status.last_fallen != 0
                    and self._world_model.timestamp - status.last_fallen >= self._state_change_delay
                ):
                    status.upright = True
