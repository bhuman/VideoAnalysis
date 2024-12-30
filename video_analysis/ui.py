from __future__ import annotations

import platform
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

if TYPE_CHECKING:
    from .world_model.team import Team


class Size(NamedTuple):
    """Tuple, which describes the width and height of an object."""

    width: int
    height: int


class UI:
    """The UI of the app.

    It is displayed when this class is instantiated. The method render must be called regularly to update the UI.
    The size of the window is dynamic. Statistics can be set through the method set_value. The state of the UI is
    stored in the settings in the groups "window" and "view".
    """

    """The number of pixels the image smaller than the window."""
    _window_to_image_size = Size(
        434 if platform.system() != "Windows" else 454, 64 if platform.system() != "Windows" else 103
    )

    _remaining_to_table_width = 35
    """Number of spare pixels after placing both tables and the image."""

    _table_to_bar_width = 140
    """Number of pixels spent in tables except for the value bars."""

    _window_to_table_height_diff = _window_to_image_size.height - 2
    """The number of pixels the tables are less high than the window."""

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
            x_pos=settings["window"]["x"],
            y_pos=settings["window"]["y"],
            width=settings["window"]["width"],
            height=settings["window"]["height"],
            min_width=self._window_to_image_size.width + 320,
            min_height=self._window_to_image_size.height + 180,
            resizable=True,
            vsync=True,
        )

        # Create placeholders for the image and field views shown in the tabs.
        image = np.zeros((1080, 1920, 4), np.uint8)
        field = np.zeros((740, 1040, 4), np.uint8)

        # Register the image and two field views as textures to be used in the three tabs.
        with dpg.texture_registry():
            dpg.add_dynamic_texture(image.shape[1], image.shape[0], image, tag="video")  # pyright: ignore[reportGeneralTypeIssues]
            dpg.add_dynamic_texture(field.shape[1], field.shape[0], field, tag="field")  # pyright: ignore[reportGeneralTypeIssues]
            dpg.add_dynamic_texture(field.shape[1], field.shape[0], field, tag="ball")  # pyright: ignore[reportGeneralTypeIssues]
            dpg.add_dynamic_texture(field.shape[1], field.shape[0], field, tag="left")  # pyright: ignore[reportGeneralTypeIssues]
            dpg.add_dynamic_texture(field.shape[1], field.shape[0], field, tag="right")  # pyright: ignore[reportGeneralTypeIssues]
            dpg.add_dynamic_texture(image.shape[1], image.shape[0], image, tag="green")  # pyright: ignore[reportGeneralTypeIssues]
            dpg.add_dynamic_texture(image.shape[1], image.shape[0], image, tag="white")  # pyright: ignore[reportGeneralTypeIssues]

        # Create the whole ui (menu, team 0 left, tabs center, team 1 right).
        with dpg.window(
            no_title_bar=True, tag="main", width=settings["window"]["width"], height=settings["window"]["height"]
        ):
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
                    dpg.add_table_column()
                    with dpg.table_row():
                        with dpg.tab_bar(callback=lambda sender, data, _: self._tab_changed(data), tag="tab"):
                            with dpg.tab(label="Video", tag="Video"):
                                dpg.add_image("video", tag="video_image")
                            with dpg.tab(label="Field", tag="Field"):
                                with dpg.group(horizontal=True, horizontal_spacing=0, tag="field_group"):
                                    dpg.add_image("field", tag="field_image")
                                    dpg.add_spacer(tag="field_padding")
                            with dpg.tab(label="Ball", tag="Ball"):
                                with dpg.group(horizontal=True, horizontal_spacing=0, tag="ball_group"):
                                    dpg.add_image("ball", tag="ball_image")
                                    dpg.add_spacer(tag="ball_padding")
                            with dpg.tab(label="Left", tag="Left"):
                                with dpg.group(horizontal=True, horizontal_spacing=0, tag="left_group"):
                                    dpg.add_image("left", tag="left_image")
                                    dpg.add_spacer(tag="left_padding")
                            with dpg.tab(label="Right", tag="Right"):
                                with dpg.group(horizontal=True, horizontal_spacing=0, tag="right_group"):
                                    dpg.add_image("right", tag="right_image")
                                    dpg.add_spacer(tag="right_padding")
                            with dpg.tab(label="Green", tag="Green"):
                                dpg.add_image("green", tag="green_image")
                            with dpg.tab(label="White", tag="White"):
                                dpg.add_image("white", tag="white_image")
                        dpg.set_value("tab", self._settings["view"]["tab"])
                self._team_info(1)
        dpg.set_primary_window("main", True)

        # Register resize handler.
        with dpg.item_handler_registry(tag="main_handler"):
            dpg.add_item_resize_handler(callback=lambda: self._size_changed())
        dpg.bind_item_handler_registry("main", "main_handler")

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
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        image = cv2.normalize(
            image,
            None,  # pyright: ignore[reportGeneralTypeIssues] # Wrong stubs.
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,  # pyright: ignore[reportGeneralTypeIssues] # Wrong stubs.
        )
        dpg.set_value(type_, image)

    def set_field(self, type_: str, field) -> None:
        """Replace the field drawing with a new one.

        :param type_: The type of the view to replace ("field", "ball", "left", or "right").
        :param field: The new field drawing.
        """
        field = cv2.cvtColor(field, cv2.COLOR_RGB2BGRA)
        field = cv2.normalize(
            field,
            None,  # pyright: ignore[reportGeneralTypeIssues] # Wrong stubs.
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,  # pyright: ignore[reportGeneralTypeIssues] # Wrong stubs.
        )
        dpg.set_value(type_, field)

    def save_state(self):
        """Save window state in settings."""
        self._settings["window"]["x"], self._settings["window"]["y"] = dpg.get_viewport_pos()
        self._settings["window"]["width"] = dpg.get_viewport_width()
        self._settings["window"]["height"] = dpg.get_viewport_height()

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
            with dpg.child_window(border=False, tag="team" + str(team), no_scrollbar=True):
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit, no_host_extendX=True):
                    dpg.add_table_column()
                    dpg.add_table_column()
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
        width = dpg.get_item_width(tag)
        assert width is not None
        dpg.delete_item(tag)
        dpg.add_progress_bar(default_value=value, overlay=overlay, width=width, tag=tag, before="temp")
        dpg.delete_item("temp")

    def _tab_changed(self, tab: str) -> None:
        """This method is called when the current tab changes.

        :param tab: The name of the tab that was activated.
        """
        if isinstance(tab, str):
            self._settings["view"]["tab"] = tab
        else:
            for alias in dpg.get_aliases():
                if dpg.get_alias_id(alias) is tab:
                    self._settings["view"]["tab"] = alias

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

    def _size_changed(self):
        """This method is called when the size of the main window changes."""
        # Compute all sizes
        width, height = dpg.get_item_rect_size("main")
        image_width = width - self._window_to_image_size.width
        image_height = min(height - self._window_to_image_size.height, image_width * 9 / 16)
        image_width = min(image_width, image_height * 16 / 9)
        field_width = image_height * 4 / 3
        left_padding = (int(image_width) - int(field_width)) // 2
        right_padding = int(image_width) - int(field_width) - left_padding
        table_width = int(width - image_width - self._remaining_to_table_width) // 2
        table_height = height - self._window_to_table_height_diff
        bar_width = int(table_width - self._table_to_bar_width)

        # Delete old images
        dpg.delete_item("video_image")
        dpg.delete_item("field_image")
        dpg.delete_item("ball_image")
        dpg.delete_item("left_image")
        dpg.delete_item("right_image")
        dpg.delete_item("green_image")
        dpg.delete_item("white_image")

        # Replace them with new images
        dpg.add_image("video", parent="Video", width=int(image_width), height=int(image_height), tag="video_image")
        dpg.add_image(
            "field",
            parent="field_group",
            width=int(field_width),
            height=int(image_height),
            indent=left_padding,
            before="field_padding",
            tag="field_image",
        )
        dpg.add_image(
            "ball",
            parent="ball_group",
            width=int(field_width),
            height=int(image_height),
            indent=left_padding,
            before="ball_padding",
            tag="ball_image",
        )
        dpg.add_image(
            "left",
            parent="left_group",
            width=int(field_width),
            height=int(image_height),
            indent=left_padding,
            before="left_padding",
            tag="left_image",
        )
        dpg.add_image(
            "right",
            parent="right_group",
            width=int(field_width),
            height=int(image_height),
            indent=left_padding,
            before="right_padding",
            tag="right_image",
        )
        dpg.add_image("green", parent="Green", width=int(image_width), height=int(image_height), tag="green_image")
        dpg.add_image("white", parent="White", width=int(image_width), height=int(image_height), tag="white_image")

        # Adapt a few sizes
        dpg.set_item_width("field_padding", right_padding)
        dpg.set_item_width("ball_padding", right_padding)
        dpg.set_item_width("left_padding", right_padding)
        dpg.set_item_width("right_padding", right_padding)
        dpg.set_item_width("team0", table_width)
        dpg.set_item_height("team0", table_height)
        dpg.set_item_width("team1", table_width)
        dpg.set_item_height("team1", table_height)
        dpg.set_item_width("Goal for0", bar_width)
        dpg.set_item_width("Goal for1", bar_width)
