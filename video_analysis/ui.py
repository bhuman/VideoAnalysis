from __future__ import annotations

import platform
from copy import deepcopy
from typing import Any, NamedTuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from .world_model.game_state import Team


class Size(NamedTuple):
    """Tuple, which describes the width and height of an object."""

    width: int
    height: int


class UI:
    """The UI of the app.

    It is displayed when this class is instantiated. The method render must be called regularly to update the UI.
    The size of the window is hardcoded. Statistics can be set through the method set_value. Look at the source code
    below to get a list of the categories that can be used. The state of the UI is stored in the settings in the
    group "view".
    """

    _window_size = Size(
        1280 if platform.system() != "Windows" else 1300, 541 if platform.system() != "Windows" else 580
    )
    """The size of the main window in pixels."""

    _image_size = Size(846, 476)
    """The size of area that shows the images from the game in pixels."""

    _field_size = Size(670, 476)
    """The size of the field views in pixels. The height should be the same as in _image_size."""

    _column_width = 60
    """The width of the progress bars."""

    def __init__(self, categories: dict[str, list], teams: tuple[Team, Team], settings: dict[str, Any]) -> None:
        """Constructor. Sets up the window.

        :param teams: The left team and the right team.
        """
        self._categories: dict[str, list] = categories
        self._shown_categories = deepcopy(categories)
        self._teams = teams
        self._settings = settings

        # Create the context and main window.
        dpg.create_context()
        dpg.create_viewport(
            title="B-Human's Video Analysis",
            x_pos=0,
            y_pos=0,
            width=self._window_size.width,
            height=self._window_size.height,
            resizable=False,
            vsync=True,
        )

        # Create placeholders for the image and field views shown in the tabs.
        image = np.zeros((self._image_size.height, self._image_size.width, 4), np.uint8)
        field = np.zeros((self._field_size.height, self._field_size.width, 4), np.uint8)

        # Register the image and two field views as textures to be used in the three tabs.
        with dpg.texture_registry():
            dpg.add_dynamic_texture(image.shape[1], image.shape[0], image, tag="video")  # pyright: ignore
            dpg.add_dynamic_texture(field.shape[1], field.shape[0], field, tag="field")  # pyright: ignore
            dpg.add_dynamic_texture(field.shape[1], field.shape[0], field, tag="heatmap")  # pyright: ignore
            dpg.add_dynamic_texture(image.shape[1], image.shape[0], image, tag="green")  # pyright: ignore
            dpg.add_dynamic_texture(image.shape[1], image.shape[0], image, tag="white")  # pyright: ignore

        # Create the whole ui (menu, team 0 left, tabs center, team 1 right).
        with dpg.window(no_title_bar=True, tag="main", width=self._window_size.width, height=self._window_size.height):
            with dpg.menu_bar():
                with dpg.menu(label="View"):
                    dpg.add_menu_item(
                        label="Background",
                        check=True,
                        default_value=self._settings["view"]["background"],
                        callback=lambda sender, data, _: self._background_changed(data),
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="None",
                        check=True,
                        default_value=not self._settings["view"]["boxes"],
                        callback=lambda sender, data, _: self._boxes_changed(sender),
                        tag="no_boxes",
                    )
                    dpg.add_menu_item(
                        label="Boxes",
                        check=True,
                        default_value=self._settings["view"]["boxes"] and not self._settings["view"]["labels"],
                        callback=lambda sender, data, _: self._boxes_changed(sender),
                        tag="boxes",
                    )
                    dpg.add_menu_item(
                        label="Boxes and Labels",
                        check=True,
                        default_value=self._settings["view"]["labels"],
                        callback=lambda sender, data, _: self._boxes_changed(sender),
                        tag="labels",
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Field Lines",
                        check=True,
                        default_value=self._settings["view"]["field_lines"],
                        callback=lambda sender, data, _: self._field_lines_changed(data),
                    )
                    dpg.add_menu_item(
                        label="Controlled Areas",
                        check=True,
                        default_value=self._settings["view"]["controlled_areas"],
                        callback=lambda sender, data, _: self._controlled_areas_changed(data),
                    )

                with dpg.menu(label="Green"):
                    self._add_color_picker("green", "min")
                    self._add_color_picker("green", "max")

                with dpg.menu(label="White"):
                    self._add_color_picker("white", "min")
                    self._add_color_picker("white", "max")

            with dpg.group(horizontal=True):
                self._team_info(0)
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit, no_host_extendX=True):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=self._image_size.width)
                    with dpg.table_row():
                        with dpg.tab_bar(callback=lambda sender, data, _: self._tab_changed(data), tag="tab"):
                            with dpg.tab(label="Video", tag="Video"):
                                dpg.add_image("video")
                            with dpg.tab(label="Field", tag="Field"):
                                dpg.add_image("field", indent=(self._image_size.width - self._field_size.width) // 2)
                            with dpg.tab(label="Heat Map", tag="Heat Map"):
                                dpg.add_image("heatmap", indent=(self._image_size.width - self._field_size.width) // 2)
                            with dpg.tab(label="Green", tag="Green"):
                                dpg.add_image("green")
                            with dpg.tab(label="White", tag="White"):
                                dpg.add_image("white")
                        dpg.set_value("tab", self._settings["view"]["tab"])
                self._team_info(1)
        dpg.set_primary_window("main", True)

        # Show the UI.
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def __del__(self) -> None:
        """Destructor. Frees the dearpygui context."""
        dpg.destroy_context()

    def render(self) -> bool:
        """Render the window. Must be called regularly.

        :return: Should the program continue to run?
        """
        # Update shown statistics categories
        for category, values in self._categories.items():
            shown_values = self._shown_categories[category]
            if len(values) >= 2:
                for team in range(0, 2):
                    if values[team] != shown_values[team]:
                        self._set_value(category, team, values[team])

        dpg.render_dearpygui_frame()
        return dpg.is_dearpygui_running()

    def set_image(self, type_: str, image) -> None:
        """Replace the image in the video tab with a new one.

        :param type: The type of the view to replace ("video", "green", or "white")".
        :param image: The new image.
        """
        image = cv2.resize(
            image, (0, 0), fx=self._image_size.width / image.shape[1], fy=self._image_size.height / image.shape[0]
        )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        image = cv2.normalize(
            image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F  # pyright: ignore # Wrong stubs.
        )
        dpg.set_value(type_, image)

    def set_field(self, type_: str, field) -> None:
        """Replace the field drawing with a new one.

        :param type_: The type of the view to replace ("field" or "heatmap").
        :param field: The new field drawing.
        """
        field = cv2.resize(
            field, (0, 0), fx=self._field_size.width / field.shape[1], fy=self._field_size.height / field.shape[0]
        )
        field = cv2.cvtColor(field, cv2.COLOR_RGB2BGRA)
        field = cv2.normalize(
            field, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F  # pyright: ignore # Wrong stubs.
        )
        dpg.set_value(type_, field)

    def _add_color_picker(self, color: str, bound: str):
        """Add a HSV color picker.

        :param color: The name of the color to read from the settings ("green" or "white").
        :param bound: Either "min" or "max".
        """
        hsv: list[int] = self._settings[color + "_mask"][bound].copy()
        hsv[0] = min(179, hsv[0])
        rgb: list[int] = cv2.cvtColor(np.array([[hsv]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0].tolist()
        dpg.add_color_picker(
            rgb,
            label=bound,
            no_alpha=True,
            no_side_preview=True,
            display_hsv=True,
            no_small_preview=True,
            callback=lambda sender, data, _: self._mask_changed(sender, data),
            tag=color + "_" + bound,
        )

    def _set_value(self, category: str, team: int, value: int | float) -> None:
        """Replace a value with a new one. Both progress bars of the category are updated.

        :param category: The category of the value.
        :param team: The number of the team the value belongs to.
        :param value: The new value.
        """
        self._shown_categories[category][team] = value
        sum_ = self._categories[category][0] + self._categories[category][1]
        self._set_progress(
            category + str(team),
            self._categories[category][team] / sum_ if sum_ > 0 else 0,
            str(self._categories[category][team])
            + ("" if len(self._categories[category]) < 3 else self._categories[category][2]),
        )
        dpg.set_value(category + str(1 - team), self._categories[category][1 - team] / sum_ if sum_ > 0 else 0)

    def _team_info(self, team: int) -> None:
        """Create a table for a team.

        :param team: The number (0 = left, 1 = right) of the team the table belongs to.
        """
        with dpg.group():
            dpg.add_text(self._teams[team].name)
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit, no_host_extendX=True):
                dpg.add_table_column()
                dpg.add_table_column(width_fixed=True, init_width_or_weight=self._column_width)
                for category, values in self._categories.items():
                    if len(values) >= 2:
                        with dpg.table_row():
                            dpg.add_text((category if len(values) < 4 else values[3]) + ":")
                            dpg.add_progress_bar(
                                default_value=values[team] / (values[0] + values[1])
                                if values[0] + values[1] > 0
                                else 0,
                                overlay=str(values[team]) + ("" if len(values) < 3 else values[2]),
                                width=-1,
                                tag=category + str(team),
                            )

    @staticmethod
    def _set_progress(tag: str, value: float, overlay: str) -> None:
        """Update the values in a progress bar.

        dearpygui does not allow to change the overlay of a progress bar.
        Therefore, the old bar is removed and a new on is added.

        :param tag: The tag that identifies the progress bar.
        :param value: The value the bar in the range 0..1.
        :param overlay: The overlay drawn over the bar.
        """
        dpg.add_text(tag="temp", before=tag)
        dpg.delete_item(tag)
        dpg.add_progress_bar(default_value=value, overlay=overlay, width=-1, tag=tag, before="temp")
        dpg.delete_item("temp")

    def _tab_changed(self, tab: str) -> None:
        """This method is called when the current tab changes.

        :param tab: The name of the tab that was activated.
        """
        self._settings["view"]["tab"] = tab

    def _background_changed(self, checked: bool) -> None:
        """This method is called when the background menu item is selected.

        :param activated: Was the menu item checked?
        """
        self._settings["view"]["background"] = checked

    def _boxes_changed(self, sender: str) -> None:
        """This method is called when the boxes radio buttons are selected in the menu.

        :param sender: The menu item that was selected.
        """
        self._settings["view"]["boxes"] = sender != "no_boxes"
        self._settings["view"]["labels"] = sender == "labels"
        dpg.set_value("no_boxes", sender == "no_boxes")
        dpg.set_value("boxes", sender == "boxes")
        dpg.set_value("labels", sender == "labels")

    def _field_lines_changed(self, checked: bool) -> None:
        """This method is called when the field lines menu item is selected.

        :param activated: Was the menu item checked?
        """
        self._settings["view"]["field_lines"] = checked

    def _controlled_areas_changed(self, checked: bool) -> None:
        """This method is called when the controlled areas menu item is selected.

        :param activated: Was the menu item checked?
        """
        self._settings["view"]["controlled_areas"] = checked

    def _mask_changed(self, sender: str, rgb: list[float]) -> None:
        """This method is called when a color picker for a mask is changed.

        :param sender: The tag of the picker that encodes both color name and bound to update.
        :param rgb: The changed color value in RGB (0..1).
        """
        hsv = cv2.cvtColor(np.array([[[v * 255 for v in rgb]]], np.uint8), cv2.COLOR_RGB2HSV)[0][0].tolist()

        # In RGB, hue = 180 is represented as hue = 0. Assume that the maximum is more likely to be 180.
        if hsv[0] == 0 and sender[6:] == "max":
            hsv[0] = 180
        self._settings[sender[:5] + "_mask"][sender[6:]] = hsv
