from __future__ import annotations

import statistics
from collections import deque
from collections.abc import Iterator
from enum import Enum
from typing import TYPE_CHECKING, Literal

import cairo
import cv2
import motrackers
import numpy as np
import numpy.typing as npt

from ..camera import Camera

if TYPE_CHECKING:
    from . import WorldModel


class Player:
    """Information about a player.

    Each player has an id assigned by the multi object tracker. It has a color history,
    a position on the field, a timestamp when it was last seen, the bounding box of its
    last detection provided by YOLOv5, and whether it is assumed to be upright. The
    color history contains recent colors the player was detected as.
    """

    class Color(Enum):
        """The team color as returned by the YOLOv5 network."""

        UNKNOWN = 0
        BLUE = 1
        RED = 2
        YELLOW = 3
        BLACK = 4
        GREEN = 5
        GRAY = 6

    def __init__(
        self,
        id_: int,
        colors: list[Player.Color],
        position: npt.NDArray[np.float_],
        last_seen: float,
        last_bb_in_image: npt.NDArray[np.float_],
        upright: bool,
        color_memory: int,
    ) -> None:
        """Initialize the player.

        :param id: The id returned by the multi object tracker.
        :param colors: The colors detected so far. This list has either one or no
        entries, depending on whether the color detected is valid for the current game.
        :param position: The position in field coordinates in m.
        :param last_seen: When the player was last seen in seconds since the start of
        the video.
        :param last_bb_in_image: The bounding box in image coordinates in the format
        `[x_min, y_min, x_max, y_max]`.
        :param upright: Is the player upright?
        :param color_memory: The number of recent color assignments to remember.
        """
        self.id_ = id_
        self.colors = deque(colors, maxlen=color_memory)
        self.position = position
        self.last_seen = last_seen
        self.last_bb_in_image = last_bb_in_image
        self.upright = upright

    def draw_on_field(self, context: cairo.Context) -> None:
        """Draw the player onto the field.

        :param context: The context to draw to.
        """
        context.save()
        context.set_source_rgb(*self.color_triple(range01=True))
        context.move_to(self.position[0], self.position[1])
        context.arc(self.position[0], self.position[1], 0.15, 0, 2 * np.pi)
        context.fill()
        context.restore()

    def draw_on_image(self, image: npt.NDArray[np.uint8], camera: Camera) -> None:
        """Draw the player onto the image.

        :param image: The image to draw onto.
        :param camera: Used to transform image coordinates to field coordinates.
        """
        radius = 0.17 if self.upright else 0.34
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        points_in_world = self.position + np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius
        points_in_image = camera.world2image(points_in_world)
        color = self.color_triple(bgr=True)
        cv2.polylines(image, [points_in_image.astype(np.int32)], True, color, 3)

    @property
    def color(self) -> Player.Color:
        """Return the color that was most often assigned to this player recently.

        :return: The color. It is either `UNKNOWN` or one of the two colors currently
        playing.
        """
        return Player.Color.UNKNOWN if len(self.colors) == 0 else statistics.mode(self.colors)

    def color_triple(self, bgr: bool = False, range01: bool = False) -> tuple[int | float, ...]:
        """Return the draw color that corresponds to the color that was most often
        assigned to this player recently.

        :param bgr: Return BGR instead of RGB.
        :param range01: Color values are between 0 and 1 instead of 0 and 255.
        :return: A tuple of three channel intensities.
        """
        color: tuple[int | float, ...] = [
            (0, 0, 0),
            (0, 128, 255),
            (255, 0, 0),
            (255, 255, 0),
            (2, 2, 2),
            (0, 255, 0),
            (192, 192, 192),
        ][self.color.value]
        if bgr:
            color = color[::-1]
        if range01:
            color = tuple(x / 255 for x in color)
        return color


class Players:
    """The players model.

    This is a very simple model. Only players that are currently visible are
    represented in the model. However, players that are not seen are remembered
    for a while, so that they keep their color assignment. A centroid multi
    object tracker is used to keep the identity of players between frames. The
    players' positions on the field are determined by projecting the centers of
    their bounding boxes into the world and assuming that these points are 26 cm
    above ground. A player is considered to be upright if the projection of its
    foot point back into the image is inside its bounding box.
    """

    _player_lost_delay = 5  # 5 seconds
    """The number of second after which an unseen player is finally dropped."""

    _color_memory_duration = 20  # 20 seconds
    """The number of seconds recent color assignments are remembered for each player."""

    _border_tolerance = 3  # 3 pixels
    """The minimum distance from the image border in which a fall detection is still applied."""

    _upright_center_above_ground = 0.26
    """The assumed height of the center of an upright player above ground."""

    _fallen_center_above_ground = 0.08
    """The assumed height of the center of a fallen player above ground."""

    _names_to_colors: dict[str, Player.Color] = {
        "Blue": Player.Color.BLUE,
        "Red": Player.Color.RED,
        "Yellow": Player.Color.YELLOW,
        "Black": Player.Color.BLACK,
        "Green": Player.Color.GREEN,
        "Gray": Player.Color.GRAY,
    }
    """The mapping from GameController color names to YOLOv5 color classes."""

    def __init__(self, world_model: WorldModel) -> None:
        """Initialize the players model.

        :param world_model: The world model.
        """
        self._world_model = world_model
        self.players: list[Player] = []
        self._remembered_players: list[Player] = []
        self._tracker = motrackers.CentroidTracker(max_lost=self._player_lost_delay * world_model.camera.fps)
        self._team_colors: list[Player.Color] = [
            self.name_to_color(team.color) for team in world_model.game_state.teams
        ]

    def update(self, percepts: list[npt.NDArray[np.float_]]) -> None:
        """Update the players model.

        :param detections: The detections made by YOLOv5. It is a list of bounding boxes
        in the format `[x_min, y_min, x_max, y_max, confidence, class (Player.Color)]`.
        """
        self.players += self._remembered_players
        tracker_boxes: list[tuple[float, float, float, float]] = []
        tracker_scores: list[float] = []
        tracker_ids: list[float] = []
        for obj in percepts:
            tracker_boxes.append((obj[0], obj[1], obj[2] - obj[0], obj[3] - obj[1]))
            tracker_scores.append(obj[4])
            tracker_ids.append(obj[5])

        # TODO: We want to get rid of this tracker library asap.
        tracks: list[tuple[int | float, ...]] = self._tracker.update(tracker_boxes, tracker_scores, tracker_ids)

        for track in tracks:
            # Unfortunately the tracker throws the class_id away, so we have to find it again.
            for obj in percepts:
                if (
                    obj[0] == track[2]
                    and obj[1] == track[3]
                    and obj[2] - obj[0] == track[4]
                    and obj[3] - obj[1] == track[5]
                ):
                    mapped_color = Player.Color(int(obj[5]))
                    position = self._world_model.camera.image2world(
                        np.array([(obj[0] + obj[2]) * 0.5, (obj[1] + obj[3]) * 0.5]), self._upright_center_above_ground
                    )
                    upright: bool = True
                    in_image = self._world_model.camera.world2image(position)
                    assert self._world_model.camera.intrinsics
                    if (
                        obj[0] > self._border_tolerance
                        and obj[2] < self._world_model.camera.intrinsics.resolution[0] - self._border_tolerance
                        and obj[1] > self._border_tolerance
                        and obj[3] < self._world_model.camera.intrinsics.resolution[1] - self._border_tolerance
                        and in_image[0] <= obj[0]
                        or in_image[0] >= obj[2]
                        or in_image[1] <= obj[1]
                        or in_image[1] >= obj[3]
                    ):
                        position = self._world_model.camera.image2world(
                            np.array([(obj[0] + obj[2]) * 0.5, (obj[1] + obj[3]) * 0.5]),
                            self._fallen_center_above_ground,
                        )
                        upright = False

                    for player in self.players:
                        if player.id_ == track[1]:
                            if mapped_color in self._team_colors:
                                player.colors.append(mapped_color)
                            player.position = position
                            player.last_seen = self._world_model.timestamp
                            player.last_bb_in_image[:] = obj[:4]
                            player.upright = upright
                            break
                    else:
                        self.players.append(
                            Player(
                                id_=int(track[1]),
                                colors=[mapped_color] if mapped_color in self._team_colors else [],
                                position=position,
                                last_seen=self._world_model.timestamp,
                                last_bb_in_image=obj[:4],
                                upright=upright,
                                color_memory=int(1 + self._world_model.camera.fps * self._color_memory_duration + 0.99),
                            )
                        )
                    break
            else:
                assert False

        # Make all players without a valid color invisible.
        for player in self.players:
            if player.last_seen == self._world_model.timestamp and player.color == Player.Color.UNKNOWN:
                player.last_seen -= 0.001

        # Remember players that are currently not seen, but still have been seen recently.
        self._remembered_players = [
            player
            for player in self.players
            if player.last_seen < self._world_model.timestamp
            and self._world_model.timestamp - player.last_seen < self._player_lost_delay
        ]

        # However, the model only contains players that are currently seen.
        self.players = [player for player in self.players if player.last_seen == self._world_model.timestamp]

    def name_to_color(self, name: str) -> Player.Color:
        """Determine the color constant for a GameController color name.

        :param name: The name of the color as used in GameController logs.
        :return: The color constant that also matches the YOLOv5 detection class.
        """
        return self._names_to_colors.get(name, Player.Color.UNKNOWN)

    def color_to_team(self, color: Player.Color) -> Literal[0, 1]:
        """Determine the team index to which a color belongs.

        :param color: The color of the team.
        :return: The index of the team. 0 plays on the left side of the video, 1
        on the right side.
        """
        return 0 if color == self._team_colors[0] else 1

    def draw_on_field(self, context: cairo.Context) -> None:
        """Draw the players onto the field.

        :param context: The context to draw to.
        """
        for player in self.players:
            player.draw_on_field(context)

    def draw_on_image(self, image: npt.NDArray[np.uint8]) -> None:
        """Draw the players onto the image.

        :param image: The image to draw onto.
        """
        for player in self.players:
            player.draw_on_image(image, self._world_model.camera)

    def __iter__(self) -> Iterator[Player]:
        """Iterator for all players currently seen."""
        return iter(self.players)
