from __future__ import annotations

import json
import os
from math import ceil
from typing import TYPE_CHECKING

import cairo
import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from . import WorldModel


class Field:
    """The field model, i.e. the dimensions of the field."""

    _max_line_length = 2
    """The maximum length of a single line drawn. Longer lines are split."""

    def __init__(self, path: os.PathLike | str, world_model: WorldModel) -> None:
        """Initialize the field model by loading it from a file.

        :param path: The path to the JSON file that contains the field dimensions.
        The file uses the format that was defined in section 4.8 of the 2021 SPL rule book,
        modified by removing the word "Box".
        """
        self._world_model: WorldModel = world_model
        with open(path, encoding="UTF-8") as file:
            dimensions = json.load(file)
            self.field_length = dimensions["field"]["length"]
            self.field_width = dimensions["field"]["width"]
            self.line_width = 0.05
            self.penalty_cross_size = dimensions["field"]["penaltyCrossSize"]
            self.penalty_area_length = dimensions["field"]["penaltyAreaLength"]
            self.penalty_area_width = dimensions["field"]["penaltyAreaWidth"]
            self.penalty_cross_distance = dimensions["field"]["penaltyCrossDistance"]
            self.has_goal_area = "goalAreaLength" in dimensions["field"]
            if self.has_goal_area:
                self.goal_area_length = dimensions["field"]["goalAreaLength"]
                self.goal_area_width = dimensions["field"]["goalAreaWidth"]
            self.center_circle_diameter = dimensions["field"]["centerCircleDiameter"]
            self.border_strip_width = dimensions["field"]["borderStripWidth"]
            self.goal_depth = dimensions["goal"]["depth"]
            self.goal_inner_width = dimensions["goal"]["innerWidth"]
            self.goal_post_diameter = dimensions["goal"]["postDiameter"]
            self.goal_height = dimensions["goal"]["height"]

        l_2 = self.field_length * 0.5
        w_2 = self.field_width * 0.5

        def pg_area(length: float, width: float, sign: int) -> list[tuple[float, float]]:
            return [
                (sign * l_2, -width * 0.5),
                (sign * (l_2 - length), -width * 0.5),
                (sign * (l_2 - length), width * 0.5),
                (sign * l_2, width * 0.5),
            ]

        def penalty_mark(sign: int) -> list[list[tuple[float, float]]]:
            return [
                [
                    (sign * (l_2 - self.penalty_cross_distance) - self.penalty_cross_size * 0.5, 0),
                    (sign * (l_2 - self.penalty_cross_distance) + self.penalty_cross_size * 0.5, 0),
                ],
                [
                    (sign * (l_2 - self.penalty_cross_distance), -self.penalty_cross_size * 0.5),
                    (sign * (l_2 - self.penalty_cross_distance), self.penalty_cross_size * 0.5),
                ],
            ]

        self._lines: list[list[tuple[float, float]]] = [
            [(-l_2, -w_2), (-l_2, w_2), (l_2, w_2), (l_2, -w_2), (-l_2, -w_2)],  # Outer field lines
            [(0, -w_2), (0, w_2)],  # Halfway line
            pg_area(self.penalty_area_length, self.penalty_area_width, -1),  # Left penalty area
            pg_area(self.penalty_area_length, self.penalty_area_width, 1),  # Right penalty area
            [(-self.penalty_cross_size * 0.5, 0), (self.penalty_cross_size * 0.5, 0)],  # Center cross
        ]

        self._lines += penalty_mark(-1)  # Left penalty mark
        self._lines += penalty_mark(1)  # Right penalty mark

        angles = np.linspace(0, 2 * np.pi, 32, endpoint=True)
        self._lines.append(
            (np.stack([np.cos(angles), np.sin(angles)], axis=-1) * self.center_circle_diameter / 2).tolist()
        )

        if self.has_goal_area:
            self._lines.append(pg_area(self.goal_area_length, self.goal_area_width, -1))  # Left goal area
            self._lines.append(pg_area(self.goal_area_length, self.goal_area_width, 1))  # Right goal area

        self._lines_in_image: list[npt.NDArray[np.float_]] | None = None

    def draw_on_field(self, context: cairo.Context) -> None:
        """Draw the field.

        :param context: The context to draw to.
        """

        # Green background
        context.set_source_rgb(0, 0.5, 0.125)
        context.paint()

        # Settings for all field markings
        context.set_line_width(self.line_width)
        context.set_source_rgb(1, 1, 1)

        for polyline in self._lines:
            context.move_to(polyline[0][0], polyline[0][1])
            for i in range(1, len(polyline)):
                context.line_to(polyline[i][0], polyline[i][1])

        # Actually draw the field markings
        context.stroke()

        l_2 = self.field_length * 0.5
        gp_r = self.goal_post_diameter * 0.5
        gp_x = l_2 - self.line_width * 0.5 + gp_r
        gp_y = (self.goal_inner_width + self.goal_post_diameter) * 0.5

        # Goal nets
        context.set_line_width(0.01)
        context.set_source_rgb(1, 1, 1)
        for x in np.linspace(l_2, l_2 + self.goal_depth, 6)[1:-1]:
            context.move_to(-x, -gp_y)
            context.line_to(-x, gp_y)
            context.move_to(x, -gp_y)
            context.line_to(x, gp_y)
        for y in np.linspace(-gp_y, gp_y, 17)[1:-1]:
            context.move_to(-(l_2 + self.goal_depth), y)
            context.line_to(-gp_x, y)
            context.move_to(l_2 + self.goal_depth, y)
            context.line_to(gp_x, y)
        context.stroke()

        # Goals
        def draw_goal(sign: int) -> None:
            # Goal posts
            context.arc(sign * gp_x, gp_y, gp_r, 0, 2 * np.pi)
            context.fill()
            context.arc(sign * gp_x, -gp_y, gp_r, 0, 2 * np.pi)
            context.fill()

            # Crossbar
            context.set_line_width(self.goal_post_diameter)
            context.move_to(sign * gp_x, -gp_y)
            context.line_to(sign * gp_x, gp_y)
            context.stroke()

            # Frame
            context.set_line_width(0.05)
            context.move_to(sign * gp_x, gp_y)
            context.line_to(sign * (l_2 + self.goal_depth), gp_y)
            context.line_to(sign * (l_2 + self.goal_depth), -gp_y)
            context.line_to(sign * gp_x, -gp_y)
            context.stroke()

        context.set_source_rgb(0.4, 0.4, 0.4)
        draw_goal(-1)
        draw_goal(1)

    def draw_on_image(self, image: npt.NDArray[np.uint8]) -> None:
        """Draw the field lines onto the image.

        :param image: The image to draw onto.
        """
        if self._lines_in_image is None:
            self._lines_in_image = []
            for polyline in self._lines:
                self._lines_in_image.append(self._world_model.camera.world2image(self._split(np.array(polyline))))

        assert self._lines_in_image is not None
        for polyline_in_image in self._lines_in_image:
            cv2.polylines(image, [polyline_in_image.astype(np.int32)], False, (2, 2, 2), 2)

    def _split(self, points: npt.NDArray[np.float32]):
        result = []
        if len(points) > 0:
            result.append(points[0])
            for i in range(1, len(points) + 1):
                point = points[i % len(points)]
                last_point = result[-1]
                offset = point - last_point
                length = np.linalg.norm(offset)
                steps = int(ceil(length / self._max_line_length))
                for j in range(1, steps + 1):
                    result.append(last_point + offset * j / steps)
        return np.asarray(result)
