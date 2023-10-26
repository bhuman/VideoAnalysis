from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cairo
import numpy as np
import numpy.typing as npt

from ..camera import Camera
from .ball import Ball
from .field import Field
from .game_state import GameState
from .players import Players

ROOT = Path(__file__).resolve().parents[2]  # 2 folder up.


class WorldModel:
    """The world model containing the ball, the players, the game state and the field model.

    `timestamp` represents the number of seconds since the beginning of the video.
    It also contains a reference to the camera, because it is required by many computations.
    """

    def __init__(
        self,
        camera: Camera,
        log: os.PathLike | str,
        snd_half: bool,
        field: os.PathLike | str | None,
        settings: dict[str, Any],
    ) -> None:
        """Initialize the world model.

        :param camera: Information about the camera.
        :param log: The path to the GameController log that belongs to the video.
        :param snd_half: Is the video showing the second half of the game?
        :param field: The path of the JSON file that specifies the field dimensions. If
        `None`, the field dimension are automatically selected based on the year the game
        took place in.
        :param settings: The settings used to compute the world model.
        """
        self.camera: Camera = camera
        self.timestamp = -1 / camera.fps
        self.game_state: GameState = GameState(log, snd_half, settings)

        if field is None:
            if self.game_state.basename[:4] < "2020":
                field = ROOT / "config" / "field_dimensions2015.json"
            else:
                field = ROOT / "config" / "field_dimensions2020.json"

        self.field: Field = Field(field, self, settings)
        self.ball: Ball = Ball(self, settings)
        self.players: Players = Players(self, settings)

    def update(self, detections: npt.NDArray[np.float_]) -> None:
        """Update the world model based on the detection made and the progress of time.

        :param detections: The detections made by YOLOv5. It is a list of bounding boxes
        in the format `[x_min, y_min, x_max, y_max, confidence, class ]`.
        """
        self.timestamp += 1 / self.camera.fps

        ball_percepts = [obj for obj in detections if obj[5] == 0]
        player_percepts = [obj for obj in detections if obj[5] != 0]

        self.game_state.update(self.timestamp, self.camera.fps)
        self.ball.update(ball_percepts)
        self.players.update(player_percepts)

    def draw_on_field(self, context: cairo.Context) -> None:
        """Draw the world model onto the field.

        Actually, this call also draws the field itself.
        :param context: The context to draw to.
        """
        self.field.draw_on_field(context)
        self.players.draw_on_field(context)
        self.ball.draw_on_field(context)

    def draw_on_image(self, image: npt.NDArray[np.uint8]) -> None:
        """Draw the world model onto the image.

        :param image: The image to draw onto.
        """
        self.players.draw_on_image(image)
        if self.ball.last_seen == self.timestamp:
            self.ball.draw_on_image(image)
