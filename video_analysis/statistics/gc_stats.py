from __future__ import annotations

from typing import Literal

from ..world_model import WorldModel


class GCStats:
    """This class implements a provider for statistics about the game state."""

    def __init__(self, world_model: WorldModel, categories: dict[str, list]) -> None:
        """Initialize the provider for statistics about the game state.

        :param world_model: The world model.
        :param categories: The statistics categories extended by this provider.
        """
        self._world_model: WorldModel = world_model
        self._categories: dict[str, list] = categories
        self._categories.update(
            {
                "Goal for": [0, 0, "", "Goals"],
                "Goals per Minute": [0, 0, "", "  per Minute"],
                "Penalties": [0, 0],
                "Illegal Ball Contact": [0, 0, "", "Ill. Ball Contact"],
                "Player Pushing": [0, 0],
                "Illegal Motion in Set": [0, 0, "", "Ill. Motion in Set"],
                "Inactive Player": [0, 0],
                "Illegal Position": [0, 0, "", "Ill. Position"],
                "Leaving the Field": [0, 0],
                "Request for PickUp": [0, 0],
                "Pushing Free Kick": [0, 0],
                "Penalty Kick": [0, 0],
                "Goal Kick": [0, 0],
                "Kick In": [0, 0],
                "Corner Kick": [0, 0],
            }
        )

    def update(self) -> None:
        """Update the game state statistics."""
        if self._world_model.game_state.changed:
            self._handle_new_event(self._world_model.game_state.current_event)

        for team in range(2):
            self._categories["Goals per Minute"][team] = round(
                self._categories["Goal for"][team] / (max(0.001, self._world_model.game_state.playing_time)) * 60,
                2,
            )

    def _handle_new_event(self, event: str) -> None:
        """Handle a new game state event.

        :param event: The game state event as read from the GameController log.
        """
        category: str | None = None
        team: Literal[0, 1] = 0 if self._world_model.game_state.teams[0].color in event else 1

        if "Illegal Positioning" in event or "Illegal Defender" in event:
            category = "Illegal Position"
        elif "Leaving the Field" in event:
            category = "Leaving the Field"
        elif "Illegal Motion in Set" in event:
            category = "Illegal Motion in Set"
        elif "Corner Kick for" in event:
            category = "Corner Kick"
        elif "Goal for" in event:
            category = "Goal for"
        elif "Player Pushing" in event:
            category = "Player Pushing"
        elif "Inactive Player" in event:
            category = "Inactive Player"
        elif "Illegal Ball Contact" in event:
            category = "Illegal Ball Contact"
        elif "Request for PickUp" in event:
            category = "Request for PickUp"
        elif "Pushing Free Kick for" in event:
            category = "Pushing Free Kick"
        elif "Goal Free Kick for" in event or "Goal Kick for" in event:
            category = "Goal Kick"
        elif "Kick In for" in event:
            category = "Kick In"
        elif "Corner Kick for" in event:
            category = "Corner Kick"

        if category is not None:
            # Increase counter of category if found.
            self._categories[category][team] += 1

            # Update the overall counter of penalties.
            if category in (
                "Illegal Ball Contact",
                "Player Pushing",
                "Illegal Motion in Set",
                "Inactive Player",
                "Illegal Position",
                "Leaving the Field",
            ):
                self._categories["Penalties"][team] += 1
