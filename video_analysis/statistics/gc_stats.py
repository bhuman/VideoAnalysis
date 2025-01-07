from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
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
                "Motion in Standby": [0, 0],
                "Motion in Set": [0, 0],
                "Inactive Player": [0, 0],
                "Illegal Position": [0, 0],
                "Illegal Position in Set": [0, 0, "", "  in Set"],
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
        while len(self._world_model.game_state.current_actions) > 0:
            self._handle_new_event(self._world_model.game_state.current_actions.pop(0))

        for team in range(2):
            self._categories["Goals per Minute"][team] = round(
                self._categories["Goal for"][team] / (max(0.001, self._world_model.game_state.playing_time)) * 60,
                2,
            )

    def _handle_new_event(self, event: dict[str, Any]) -> None:
        """Handle a new game state event.

        :param event: The game state event as read from the GameController log.
        """
        category: str | None = None

        if event["source"] == "user":
            current_action = event["action"]
            args = current_action["args"]

            if "penalize" in current_action["type"] and "unpenalize" not in current_action["type"]:
                team: Literal[0, 1] = 0 if self._world_model.game_state.teams[0].side in args["side"] else 1
                if "illegalPosition" in args["call"]:
                    category = (
                        "Illegal Position in Set"
                        if self._world_model.game_state.state == self._world_model.game_state.State.SET
                        else "Illegal Position"
                    )
                elif "leavingTheField" in args["call"]:
                    category = "Leaving the Field"
                elif "motionInStandby" in args["call"]:
                    category = "Motion in Standby"
                elif "motionInSet" in args["call"]:
                    category = "Motion in Set"
                elif "pushing" in args["call"]:
                    category = "Player Pushing"
                elif "fallenInactive" in args["call"]:
                    category = "Inactive Player"
                elif "foul" in args["call"]:
                    category = "Pushing Free Kick"
                elif "requestForPickUp" in args["call"]:
                    category = "Request for PickUp"
                elif "penaltyKick" in args["call"]:
                    category = "Penalty Kick"
                elif "foul" in args["call"]:
                    category = "Pushing Free Kick"
                else:
                    return
            elif "startSetPlay" in current_action["type"]:
                team = 0 if self._world_model.game_state.teams[0].side in args["side"] else 1
                if "cornerKick" in args["setPlay"]:
                    category = "Corner Kick"
                elif "goalKick" in args["setPlay"]:
                    category = "Goal Kick"
                elif "kickIn" in args["setPlay"]:
                    category = "Kick In"
                else:
                    return
            elif "goal" in current_action["type"]:
                team = 0 if self._world_model.game_state.teams[0].side in args["side"] else 1
                category = "Goal for"
            else:
                return
        else:
            return

        # Increase counter of category if found.
        self._categories[category][team] += 1

        # Update the overall counter of penalties.
        if category in (
            "Illegal Ball Contact",
            "Player Pushing",
            "Motion in Standby",
            "Motion in Set",
            "Inactive Player",
            "Illegal Position",
            "Illegal Position in Set",
            "Leaving the Field",
        ):
            self._categories["Penalties"][team] += 1
