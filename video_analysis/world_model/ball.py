from __future__ import annotations

from typing import TYPE_CHECKING

import cairo
import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from . import WorldModel


class Ball:
    """The ball model.

    This is currently not much more than a mapping from image coordinates into world
    coordinates and filtering some outliers based on an assumed maximum speed of the
    ball. In particular, there is no model about the current speed of the ball.
    Therefore, if the ball is not seen, its position simply stays the same.
    """

    _max_velocity = 2  # m/s
    """The assumed maximum speed of the ball."""

    def __init__(self, world_model: WorldModel) -> None:
        """Initialize the ball model.

        :param world_model: The world model.
        """
        self._world_model = world_model
        self.position: npt.NDArray[np.float_] = np.array([0, 0], dtype=np.float32)
        self.last_seen: float = -1.0
        self.last_bb_in_image: npt.NDArray[np.float_] = np.array([0, 0, 0, 0], dtype=np.float32)

    def update(self, percepts: list[npt.NDArray[np.float_]]) -> None:
        """Update the ball model.

        :param detections: The detections made by YOLOv5. It is a list of bounding boxes
        in the format `[x_min, y_min, x_max, y_max, confidence, class (0 = ball)]`.
        """

        # Only integrate balls in SET and PLAYING. Otherwise, reset the state.
        # TODO: During penalty kicks, the ball is somewhere else.
        if self._world_model.game_state.state not in [
            self._world_model.game_state.State.SET,
            self._world_model.game_state.State.PLAYING,
        ]:
            self.position = np.array([0, 0], dtype=np.float32)
            self.last_seen = -1.0
            return

        # Return if there aren't any percepts this frame. (TODO: later, the process model should still be applied)
        if not percepts:
            return

        # TODO: handle multiple balls.
        obj = percepts[0]
        position = self._world_model.camera.image2world(
            np.array([(obj[0] + obj[2]) * 0.5, (obj[1] + obj[3]) * 0.5]), z=0.05
        )

        # Discard measurements that are outside the field or too far away from the previous model.
        if (
            abs(position[0]) > self._world_model.field.field_length / 2
            or abs(position[1]) > self._world_model.field.field_width / 2
            or self.last_seen >= 0
            and np.linalg.norm(self.position - position)
            > self._max_velocity * (self._world_model.timestamp - self.last_seen)
        ):
            return

        self.last_seen = self._world_model.timestamp
        self.last_bb_in_image[:] = obj[:4]
        self.position = position

    def draw_on_field(self, context: cairo.Context) -> None:
        """Draw the ball onto the field.

        :param context: The context to draw to.
        """
        context.save()
        context.set_source_rgb(1, 0.5, 0)
        context.move_to(self.position[0], self.position[1])
        context.arc(self.position[0], self.position[1], 0.05, 0, 2 * np.pi)
        context.fill()
        context.restore()

    def draw_on_image(self, image: npt.NDArray[np.uint8]) -> None:
        """Draw the ball onto the image.

        :param image: The image to draw onto.
        """
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        points_in_world = self.position + np.stack([np.cos(angles), np.sin(angles)], axis=-1) * 0.1
        points_in_image = self._world_model.camera.world2image(points_in_world)
        cv2.polylines(image, [points_in_image.astype(np.int32)], True, (0, 128, 255), 3)
