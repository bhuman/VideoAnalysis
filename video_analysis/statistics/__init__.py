from __future__ import annotations

import csv
import locale
from pathlib import Path
from typing import Any

import cairo
import numpy as np
import numpy.typing as npt
import scipy.ndimage as nd
from matplotlib import cm

from ..world_model import WorldModel
from .ball_isolation_time import BallIsolationTime
from .ball_movement_time import BallMovementTime
from .controlled_areas import ControlledAreas
from .distance_counter import DistanceCounter
from .falls import Falls
from .gc_stats import GCStats
from .goal_distance import GoalDistance
from .localization import Localization
from .possession import Possession

ROOT = Path(__file__).resolve().parents[2]  # 2 folder up.


class Statistics:
    """The collection of all statistics providers. Also maintains that ball heat map."""

    def __init__(self, world_model: WorldModel, settings: dict[str, Any]) -> None:
        """Initialize the statistics providers.

        Creates a number of statistics providers that add the categories they
        provide to a common dictionary.
        """
        self._world_model: WorldModel = world_model
        self._settings: dict[str, Any] = settings

        # The statistics categories.
        # The dictionary keys are the categories. The associated values
        # are lists of the values stored for the team shown left and right.
        # Optionally, a string printed after the value can be specified.
        # Also optionally, an alternate label text can be set.
        self.categories: dict[str, list] = {}

        # The statistics providers are created here. The order is important,
        # because it mostly reflects how the statistics categories appear in
        # the UI. `distance_counter` and `possession` depend on each other
        # and must be created in this order.
        self._providers: list = [
            BallIsolationTime(world_model, self.categories),
            GCStats(world_model, self.categories),
            Falls(world_model, self.categories),
            distance_counter := DistanceCounter(world_model, self.categories),
            ball_movement_time := BallMovementTime(world_model, distance_counter, self.categories),
            possession := Possession(world_model, distance_counter, ball_movement_time, self.categories),
            GoalDistance(world_model, self.categories),
            controlled_areas := ControlledAreas(world_model, self.categories),
        ]
        if self._world_model.game_state.has_player_data:
            localization: Localization | None = Localization(world_model, self.categories, settings)
            self._providers.append(localization)
            self._localization = localization
            controlled_areas.localization = localization
        else:
            self._localization = None
        self._possession: Possession = possession
        self._controlled_areas: ControlledAreas = controlled_areas
        self._heatmap: list[npt.NDArray[np.float_]] = [
            np.zeros((740, 1040)),
            np.zeros((740, 1040)),
            np.zeros((740, 1040)),
        ]

    def update(self) -> None:
        """Update all statistics and the ball heat map."""
        for provider in self._providers:
            provider.update()

        if self._world_model.game_state.state == self._world_model.game_state.State.PLAYING:
            if self._world_model.ball.last_seen == self._world_model.timestamp:
                x_idx = np.maximum(70, np.minimum(int(np.round(self._world_model.ball.position[0] * 100) + 520), 970))
                y_idx = np.maximum(70, np.minimum(int(np.round(self._world_model.ball.position[1] * 100) + 370), 670))
                self._heatmap[2][y_idx][x_idx] += 1
            for player in self._world_model.players:
                x_idx = np.maximum(70, np.minimum(int(np.round(player.position[0] * 100) + 520), 970))
                y_idx = np.maximum(70, np.minimum(int(np.round(player.position[1] * 100) + 370), 670))
                self._heatmap[self._world_model.players.color_to_team(player.color)][y_idx][x_idx] += 1

    def get_heatmap(self, index: int) -> npt.NDArray[np.uint8]:
        """Return the visual representation of the ball heat map.

        :param index: 0 = left team, 1 = right team, 2 = ball
        :return: The heat map as bitmap.
        """

        # Blur heat map.
        heatmap = np.sqrt(nd.gaussian_filter(self._heatmap[index], sigma=20))

        # Normalize it.
        maxValue = np.max(heatmap)
        if maxValue != 0.0:
            minValue = np.min(heatmap)
            heatmap = (heatmap - minValue) / (maxValue - minValue)  # type: ignore[operator]

        # Set color map.
        heatmap = np.array(cm.jet_r(heatmap, bytes=True))  # pyright: ignore # Undocumented function

        # The heat map is upside down. Flip and return it.
        return np.flip(heatmap, axis=0)

    def save(self, basename: str) -> None:
        """Save the statistics to a csv file.

        To be able to directly read it with a spreadsheet application, the decimal mark and the field
        delimiter are based on the current locale.
        """
        path = ROOT / "statistics"
        path.mkdir(parents=True, exist_ok=True)
        with path.joinpath(f"{basename}.csv").open(mode="w", encoding="UTF-8", newline="\n") as file:
            # Switch to system locale. For unknown reasons, this is initially not the case for numbers.
            saved_locale = locale.getlocale(locale.LC_NUMERIC)
            locale.setlocale(locale.LC_NUMERIC, "")

            # Use ";" as delimiter if a comma is used as decimal mark.
            writer = csv.writer(file, delimiter=";" if locale.localeconv()["decimal_point"] == "," else ",")
            writer.writerow(["category", "for team", "against team", "value"])

            # Write all categories for both teams.
            for team in range(2):
                for category, values in self.categories.items():
                    writer.writerow(
                        [
                            category,
                            self._world_model.game_state.teams[team].name,
                            self._world_model.game_state.teams[1 - team].name,
                            locale.str(values[min(len(values) - 1, team)]),
                        ]
                    )

            # Restore locale for numbers.
            locale.setlocale(locale.LC_NUMERIC, saved_locale)

    def draw_on_field(self, context: cairo.Context) -> None:
        """Draw visualization of some statistics onto the field.

        :param context: The context to draw to.
        """
        self._possession.draw_on_field(context)
        if self._settings["view"]["controlled_areas"]:
            self._controlled_areas.draw_on_field(context)
        # FIXME
        if self._localization is not None:
            self._localization.draw_on_field(context)

    def draw_on_image(self, image: npt.NDArray[np.uint8]) -> None:
        """Draw visualization of some statistics to the image.

        :param image: The image to draw onto.
        """
        self._possession.draw_on_image(image)
        if self._localization is not None:
            self._localization.draw_on_image(image)
        if self._settings["view"]["controlled_areas"]:
            self._controlled_areas.draw_on_image(image)
