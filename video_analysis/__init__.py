from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cairo
import click
import cv2
import numpy as np
import numpy.typing as npt
from click._termui_impl import ProgressBar
from torch.backends import cudnn
from yolov5.utils.plots import Annotator

from .detection import Detector
from .sources import SourceAdapter
from .statistics import Statistics
from .ui import UI
from .world_model import WorldModel

ROOT = Path(__file__).resolve().parents[1]  # 1 folder up.


class VideoAnalysis:
    """The main class of the app.

    It basically binds all parts together. It handles whether the app opens
    a window or runs in headless mode. It reads the settings in the beginning
    and writes them back when the app ends. It also initiates writing the
    statistics to a file if the video was played back completely.
    """

    def __init__(
        self,
        video: str,
        log: os.PathLike | str,
        field: os.PathLike | str | None,
        half: int,
        every_nth_frame: int,
        calibration: os.PathLike | str | None,
        force: bool,
        skip: bool,
        verbose: bool,
        weights: os.PathLike | str,
        headless: bool,
    ) -> None:
        """Create video analysis main class.

        :param video: Video file path.
        :param log: GameController log path.
        :param field: The field dimensions path.
        :param half: Half the video shows (0 -> determine from file name)
        :param every_nth_frame: Only use every nth frame.
        :param calibration: The camera calibration path.
        :param force: Force new camera calibration.
        :param skip: Skip camera calibration even if none exists.
        :param verbose: Create verbose output during camera calibration.
        :param weights: Model.pt path(s).
        """
        # Load settings from file
        self._settings_path = ROOT / "config" / "settings.json"
        with self._settings_path.open(encoding="UTF-8", newline="\n") as file:
            self._settings: dict[str, Any] = json.load(file)

        # Guess half from name of video
        if half == 0:
            filename = Path(video).name
            if "2nd" in filename or "half2" in filename:
                half = 2

        # Load model
        self._detector = Detector(
            weights,
            imgsz=[1088, 1920],
            device=self._settings["detector"]["device"],
            dnn=self._settings["detector"]["dnn"],
            half=self._settings["detector"]["half"],
            conf_thres=self._settings["detector"]["conf_thres"],
            iou_thres=self._settings["detector"]["iou_thres"],
            agnostic_nms=True,
            max_det=self._settings["detector"]["max_det"],
        )
        imgsz = self._detector.imgsz
        stride: int = self._detector.model.stride  # pyright: ignore # Typing issue.
        pt: bool = self._detector.model.pt  # pyright: ignore # Typing issue.

        # Load data again
        self._dataset = SourceAdapter(video, imgsz, stride, pt, step=every_nth_frame)
        if self._dataset.webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference

        self._world_model = WorldModel(self._dataset.fps, log, half == 2, field)
        self._world_model.camera.calibrate(
            video,
            imgsz,
            stride,
            pt,
            self._world_model.field,
            self._settings,
            ROOT / "config" / (self._world_model.game_state.basename + ".json")
            if calibration is None
            else Path(calibration),
            force,
            skip,
            verbose,
        )  # Calibrate the mapper

        # Initialize the statistics
        self._statistics = Statistics(self._world_model, self._settings)

        # Create main window if not in headless mode.
        self._ui: UI | None = (
            None if headless else UI(self._statistics.categories, self._world_model.game_state.teams, self._settings)
        )

    def run(self) -> None:
        """Run the app."""

        # Add a progress bar in headless mode.
        if self._ui is None and self._dataset.frames() > 0:
            with click.progressbar(length=self._dataset.frames()) as progress_bar:
                self._run(progress_bar)
        else:
            self._run(None)

    def _run(self, progress_bar: ProgressBar | None):
        """Actually running the app."""

        # Show a progress bar in headless mode if the number of frames is known.
        for preprocessed_image, original_image in self._dataset:
            # Run object detector
            detections = self._detector.run(preprocessed_image, original_image.shape)

            # Update the world model using this frame's detections
            self._world_model.update(detections)

            # Update statistics using the current world model
            self._statistics.update()

            if self._ui is not None:
                # Draw tabs
                self._draw(original_image)

                # Render UI and exit if close was pressed.
                if not self._ui.render():
                    break

            # Update progress bar if there is one.
            elif progress_bar is not None:
                progress_bar.update(1)

        if self._ui is None or self._ui.render():
            while self._ui is not None and self._ui.render():
                # pylint: disable-next=undefined-loop-variable
                self._draw(original_image)  # pyright: ignore # Is always present.

            # Only save statistics if video was played back completely
            self._statistics.save(self._world_model.game_state.basename)

        # Write settings back to file
        with self._settings_path.open("w", encoding="UTF-8", newline="\n") as file:
            json.dump(self._settings, file, indent=4)
            file.write("\n")  # Add newline cause py JSON does not.

    def _draw(self, image: npt.NDArray[np.uint8]) -> None:
        """Draws the main view.

        :param image: If selected view is an image view, draw onto this image.
        """
        if self._settings["view"]["tab"] == "Field" or self._settings["view"]["tab"] == "Heat Map":
            self._draw_field()
        else:
            self._draw_image(image)

    def _draw_image(self, image: npt.NDArray[np.uint8]) -> None:
        """Draws an image view.

        :param image: Draw onto this image or use it to draw segmentation masks.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        assert self._ui is not None
        if self._settings["view"]["tab"] == "White":
            background_mask = cv2.inRange(
                hsv, np.array(self._settings["white_mask"]["min"]), np.array(self._settings["white_mask"]["max"])
            )
            self._ui.set_image("white", background_mask)
            return

        background_mask = cv2.inRange(
            hsv, np.array(self._settings["green_mask"]["min"]), np.array(self._settings["green_mask"]["max"])
        )

        if self._settings["view"]["tab"] == "Green":
            self._ui.set_image("green", background_mask)
            return

        # Field lines and bounding boxes are directly drawn onto the image. They are
        # not only drawn on green areas, because they might be hidden in that case.
        if self._settings["view"]["boxes"] or self._settings["view"]["field_lines"]:
            # Do not draw on original image.
            image = image.copy()

            if self._settings["view"]["field_lines"]:
                self._world_model.field.draw_on_image(image)

            if self._settings["view"]["boxes"]:
                # Add labels to image
                annotator = Annotator(image, line_width=2)
                for player in self._world_model.players:
                    label = ""
                    if self._settings["view"]["labels"]:
                        label = f"id{player.id_} {player.color.name}"
                    annotator.box_label(player.last_bb_in_image, label, color=player.color_triple(bgr=True))
                if self._world_model.ball.last_seen == self._world_model.timestamp:
                    label = ""
                    if self._settings["view"]["labels"]:
                        label = "ball"
                    annotator.box_label(self._world_model.ball.last_bb_in_image, label, color=(0, 128, 255))

        assert self._world_model.camera.intrinsics
        overlay = np.zeros(
            (
                int(self._world_model.camera.intrinsics.resolution[1]),
                int(self._world_model.camera.intrinsics.resolution[0]),
                3,
            ),
            dtype=np.uint8,
        )
        self._world_model.draw_on_image(overlay)
        self._statistics.draw_on_image(overlay)
        overlay = cv2.bitwise_and(overlay, overlay, mask=background_mask)
        overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, overlay_mask = cv2.threshold(overlay_gray, 1, 255, cv2.THRESH_BINARY)
        overlay_mask = cv2.bitwise_not(overlay_mask)
        image = cv2.bitwise_and(image, image, mask=overlay_mask)
        image = cv2.add(image, overlay)
        self._ui.set_image("video", image)

    def _draw_field(self) -> None:
        """Draw the field or heat map."""
        # TODO: Don't hardcode this:
        width = 1040
        height = 740
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        # Draw field drawings (field origin is at center of the canvas)
        ctx.scale(
            width / (self._world_model.field.field_length + 2 * self._world_model.field.border_strip_width),
            height / (self._world_model.field.field_width + 2 * self._world_model.field.border_strip_width),
        )
        ctx.translate(
            self._world_model.field.field_length * 0.5 + self._world_model.field.border_strip_width,
            self._world_model.field.field_width * 0.5 + self._world_model.field.border_strip_width,
        )
        ctx.scale(1, -1)  # TODO: flipping y is necessary but breaks font drawings (because they are flipped as well)
        self._world_model.draw_on_field(ctx)
        self._statistics.draw_on_field(ctx)

        draw_heatmap = self._settings["view"]["tab"] == "Heat Map"
        if not draw_heatmap:
            self._statistics.draw_on_field(ctx)

        bitmap = np.reshape(
            np.frombuffer(surface.get_data(), dtype=np.uint8), (surface.get_height(), surface.get_width(), 4)
        )

        assert self._ui is not None
        if draw_heatmap:
            heatmap = self._statistics.get_heatmap()
            heatmap_on_field = cv2.addWeighted(heatmap, 0.7, bitmap, 0.3, 0)
            self._ui.set_field("heatmap", heatmap_on_field)
        else:
            self._ui.set_field("field", bitmap)
