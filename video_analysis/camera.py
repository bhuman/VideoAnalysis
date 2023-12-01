"""This module uses a lot of the code from the repository https://github.com/BerlinUnited/RoboCupTools
(some of it modified though).
It contains everything needed to calculate an aerial view of the game that has been recorded from a side view.
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R
from telemetry_parser import Parser as GPMFParser  # type: ignore[attr-defined]

from .sources import SourceAdapter
from .world_model.field import Field

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]  # 1 folder up.


@dataclass
class Intrinsics:
    """The intrinsic calibration of the camera."""

    optical_center: npt.NDArray[np.float_]
    focal_length: npt.NDArray[np.float_]
    k1: float
    k2: float
    k3: float
    superview: bool
    resolution: npt.NDArray[np.float_]

    @classmethod
    def from_vector(cls, intrinsics_vector: npt.NDArray[np.float_]):
        """Creates an object from a vector with 10 elements.

        :param intrinsics_vector: The intrinsics vector.
        :return: A new object with the given intrinsics.
        """
        return cls(
            optical_center=intrinsics_vector[0:2],
            focal_length=intrinsics_vector[2:4],
            k1=intrinsics_vector[4],
            k2=intrinsics_vector[5],
            k3=intrinsics_vector[6],
            superview=bool(intrinsics_vector[7] != 0),
            resolution=intrinsics_vector[8:10],
        )

    def to_vector(self) -> npt.NDArray[np.float_]:
        """Calculates a vector with 10 elements of this object.

        :return: The intrinsics vector.
        """
        return np.concatenate(
            [
                self.optical_center,
                self.focal_length,
                np.array([self.k1, self.k2, self.k3, 1.0 if self.superview else 0.0]),
                self.resolution,
            ]
        )

    @classmethod
    def from_dict(cls, intrinsics: dict[str, Any]):
        """Creates the intrinsics from a dictionary.

        :param intrinsics: The intrinsics as a dictionary.
        :return: A new object with the intrinsics extracted from the dictionary.
        """
        # with path.open(encoding="UTF-8", newline="\n") as file:
        return cls(
            optical_center=np.array(intrinsics["optical_center"], dtype=np.float32),
            focal_length=np.array(intrinsics["focal_length"], dtype=np.float32),
            k1=intrinsics["k1"],
            k2=intrinsics["k2"],
            k3=intrinsics["k3"],
            superview=intrinsics["superview"],
            resolution=np.array(intrinsics["resolution"], dtype=np.float32),
        )

    def to_dict(self) -> dict[str, Any]:
        """Creates a dictionary representing the intrinsics.

        :return: The intrinsics as a dictionary.
        """
        intrinsics: dict[str, Any] = {}
        intrinsics["optical_center"] = self.optical_center.tolist()
        intrinsics["focal_length"] = self.focal_length.tolist()
        intrinsics["k1"] = self.k1
        intrinsics["k2"] = self.k2
        intrinsics["k3"] = self.k3
        intrinsics["superview"] = self.superview
        intrinsics["resolution"] = self.resolution.tolist()
        return intrinsics


@dataclass
class Extrinsics:
    """The extrinsic calibration of the camera (i.e. pose of the camera in the world).

    Besides the representation of the extrinsics that is stored in this class
    (i.e. a 3x3 rotation matrix and 3D translation vector), there is also a "packed"
    vector representation as xyz-Euler angles (in degrees) concatenated with
    the translation vector. The representation stored in the JSON file is a
    combination of both, i.e. rotation and translation are stored as separate
    elements, but the rotation is represented as xyz-Euler angles instead of a
    rotation matrix.
    """

    rotation: npt.NDArray[np.float_]
    translation: npt.NDArray[np.float_]

    @classmethod
    def from_vector(cls, extrinsics_vector: npt.NDArray[np.float_]):
        """Creates an object from the 6D vector representation.

        :param extrinsics_vector: The 6D extrinsics vector.
        :return: A new object with the given extrinsics.
        """
        rotation = R.from_euler("xyz", extrinsics_vector[0:3], degrees=True).as_matrix()
        return cls(rotation=rotation, translation=extrinsics_vector[3:6])

    def to_vector(self) -> npt.NDArray[np.float_]:
        """Calculates the 6D vector representation of this object.

        :return: The 6D extrinsics vector.
        """
        angles = R.from_matrix(self.rotation).as_euler("xyz", degrees=True)
        return np.concatenate([angles, self.translation])

    @classmethod
    def from_dict(cls, extrinsics: dict[str, Any]):
        """Creates the extrinsics from a dictionary.

        :param extrinsics: The extrinsics as a dictionary.
        :return: A new object with the extrinsics extracted from the dictionary.
        """
        return cls.from_vector(np.array(extrinsics["rotation"] + extrinsics["translation"], dtype=np.float32))

    def to_dict(self) -> dict[str, Any]:
        """Creates a dictionary representing the extrinsics.

        :return: The extrinsics as a dictionary.
        """
        as_vector = self.to_vector()
        extrinsics: dict[str, Any] = {}
        extrinsics["rotation"] = as_vector[0:3].tolist()
        extrinsics["translation"] = as_vector[3:6].tolist()
        return extrinsics


class Camera:
    """Provides transformations between camera (i.e. image) coordinates and world (i.e. field)
    coordinates.

    The origin of the world coordinates is at the center point of the field. The first dimension
    points to the right. The second dimension points away from the camera. The third dimension
    points upward. World coordinates are in meters.
    This class also stores the number of frames per second at which the video is played back.
    In addition, the background image, i.e. the field without referees and robots, is computed
    and can be accessed.
    """

    def __init__(self, fps: float, settings: dict[str, Any]) -> None:
        """Initialize the mapper between image and world coordinates.

        :param fps: The number of frames per second at which the video is played back.
        :param settings: The settings used to configure the calibration process.
        """
        self.fps: float = fps
        self._settings: dict[str, Any] = settings
        self.background: cv2.Mat | None = None
        self._extrinsics: Extrinsics | None = None
        self.intrinsics: Intrinsics | None = None

    # The initial calibration of this mapper. Takes some images and calculates a mapping from points in the source video
    # to points on the football pitch.
    def calibrate(
        self,
        source: str,
        imgsz: list[int],
        stride: int,
        pt: bool,
        field: Field,
        calibration_path: Path,
        force: bool,
        verbose: bool,
        basename: str,
    ) -> None:
        """Calibrate the mapper or load a previous calibration.

        :param source: Path or URL of the video or a folder with images. Could also be the
        number of the webcam (which is currently not supported).
        :param imgsz: The height and width of the image (in that order) or a single number
        if both sizes are the same.
        :param stride: Stride of the model. Image size is increased to a multiple of this value.
        :param pt: Using PyTorch?
        :param field: The field model.
        :param settings: The settings.
        :param calibration_path: Read calibration from a file with this path or store it in it.
        :param force: Do a new calibration even if one exists.
        :param verbose: Write additional information to the terminal while calibrating and
        visualize the results under the path `runs/camera/run_*`.
        :param basename: A name that is used to create the file name for storing or loading
        background images.
        """
        background_path = (
            ROOT
            / "backgrounds"
            / f"{basename}_{self._settings['calibration']['images']}_{self._settings['calibration']['step_size']}.jpg"
        )
        if background_path.is_file():
            self.background = cv2.imread(str(background_path))

        if not force and calibration_path.is_file():
            with calibration_path.open(encoding="UTF-8", newline="\n") as file:
                calibration: dict[str, Any] = json.load(file)
                self._extrinsics = Extrinsics.from_dict(calibration["extrinsics"])
                self.intrinsics = Intrinsics.from_dict(calibration["intrinsics"])
            return

        log_level = logging.INFO
        if verbose:
            # Change logging level to print all debug messages during calibration.
            log_level = logger.level
            logger.setLevel(logging.DEBUG)
        logger.info("Calibrating camera...")
        # Load data once
        dataset = SourceAdapter(
            source, imgsz, stride, pt, step=self._settings["calibration"]["step_size"]
        )  # We only select every hundredth image for a better quality of the calibration

        if self.background is None:
            self.background = self._read_video_background(dataset, self._settings["calibration"]["images"])
            (ROOT / "backgrounds").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(background_path), self.background)

        points_original = self._detect_lines(self.background)
        try:
            points_reduced, points_transformed, points_transformed_initial = self._align_camera(
                source,
                field,
                points_original,
            )
            assert self.intrinsics is not None
            assert self._extrinsics is not None

            with calibration_path.open(mode="w", encoding="UTF-8", newline="\n") as file:
                calibration = {
                    "intrinsics": self.intrinsics.to_dict(),
                    "extrinsics": self._extrinsics.to_dict(),
                }
                json.dump(calibration, file, indent=4)
                file.write("\n")

            # The following code is helpful when tweaking the parameters.
            if verbose:
                camera_path = ROOT / "runs" / "camera"
                camera_path.mkdir(parents=True, exist_ok=True)
                run_number = 1
                while camera_path.joinpath(f"run_{run_number}").exists():
                    run_number += 1
                path = camera_path.joinpath(f"run_{run_number}")
                path.mkdir()
                _, (ax1, ax2) = plt.subplots(2, 1)  # pyright: ignore
                ax1.set_aspect("equal")
                # Here, the first parameter is for the horizontal axis, the second for the vertical one.
                ax1.plot(points_original[:, 0], self.background.shape[0] - points_original[:, 1], ".")
                ax1.set_title("Detected line points")
                ax2.set_aspect("equal")
                ax2.plot(points_reduced[:, 0], self.background.shape[0] - points_reduced[:, 1], ".")
                ax2.set_title("reduced line points")
                plt.savefig(path / "before.png")
                _, (ax3, ax4) = plt.subplots(2, 1)
                ax3.set_aspect("equal")
                ax3.plot(points_transformed[:, 0], points_transformed[:, 1], ".")
                ax3.set_title("final projection")
                ax4.set_aspect("equal")
                ax4.plot(points_transformed_initial[:, 0], points_transformed_initial[:, 1], ".")
                ax4.set_title("Initial transformation")
                plt.savefig(path / "after.png")
                logger.setLevel(log_level)
            logger.info("Calibration finished.")

        except RuntimeError:
            logger.error("Calibration failed. Using default values.")
            self._extrinsics = Extrinsics.from_dict(self._settings["calibration"]["initial_extrinsics"])
            self.intrinsics = Intrinsics.from_dict(self._settings["calibration"]["initial_intrinsics"])

    def image2world(self, points_in_image: npt.NDArray[np.float_], z: float = 0.0) -> npt.NDArray[np.float_]:
        """Transforms points from the image into the world.

        :param points_in_image: A list of points or a single point in image coordinates.
        :param z: The height of the horizontal plane above ground the points are projected to.
        :return: 2D point(s) in world (i.e. field) coordinates.
        """
        assert self.intrinsics is not None and self._extrinsics is not None
        assert points_in_image.shape[-1] == 2
        single = len(points_in_image.shape) == 1
        if single:
            points_in_image = points_in_image[np.newaxis, :]
        points_in_camera = self._image2camera(points_in_image, self.intrinsics)
        points_in_world = self._camera2world(points_in_camera, self._extrinsics, z_in_world=z)
        return points_in_world[0] if single else points_in_world

    def world2image(self, points_in_world: npt.NDArray[np.float_], z: float = 0.0) -> npt.NDArray[np.float_]:
        """Transform points in the world back into the image.

        :param points_in_world: A list of 2D points or a single point in world (i.e. field) coordinates.
        :param z: The height of the points above ground.
        :return: The point(s) in image coordinates.
        """
        assert self._extrinsics is not None and self.intrinsics is not None
        assert points_in_world.shape[-1] == 2
        single = len(points_in_world.shape) == 1
        if single:
            points_in_world = points_in_world[np.newaxis, :]
        points_in_camera = self._world2camera(points_in_world, self._extrinsics, z_in_world=z)
        points_in_image = self._camera2image(points_in_camera, self.intrinsics)
        return points_in_image[0] if single else points_in_image

    @staticmethod
    def _read_video_background(dataset: SourceAdapter, images: int) -> cv2.Mat:
        """Determine the video background from a list of images.

        This only works if foreground objects move sufficiently in the dataset.
        :param dataset: The dataset that contains the images. The original images are used.
        :param images: The number of images to use.
        :return: The background image, i.e. foreground object are mostly removed.
        """
        subtractor = cv2.createBackgroundSubtractorMOG2()
        with click.progressbar(length=images) as progress_bar:
            for i, (_, image) in enumerate(dataset):
                subtractor.apply(image)
                progress_bar.update(1)
                if i >= images:
                    break
        return subtractor.getBackgroundImage()

    @staticmethod
    def _skeleton(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Create the skeleton of a mask.

        :param mask: The mask containing the field lines.
        :return: The mask with field lines reduced to a thickness of a single pixel.
        """
        size = np.size(mask)
        skeleton_ = np.zeros(mask.shape, dtype=np.uint8)

        _, mask = cv2.threshold(mask, 127, 255, 0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(mask, temp)
            skeleton_ = cv2.bitwise_or(skeleton_, temp)
            mask = eroded.copy()

            zeros = size - cv2.countNonZero(mask)
            if zeros == size:
                done = True

        return skeleton_

    def _field_mask(self, image: cv2.Mat) -> npt.NDArray[np.uint8]:
        """Generate mask from the image that only contains the field lines.

        :param image: The image.
        """

        # Convert image to the HSV color space.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green mask
        lower_green = np.array(self._settings["green_mask"]["min"])
        upper_green = np.array(self._settings["green_mask"]["max"])
        mask_field = cv2.inRange(hsv, lower_green, upper_green)

        # Remove noise.
        mask_field = cv2.erode(mask_field, None, iterations=3)
        mask_field = cv2.dilate(mask_field, None, iterations=3)

        # Close the lines inside the field.
        mask_field = cv2.dilate(mask_field, None, iterations=20)

        # Remove green bits outside the field.
        mask_field = cv2.erode(mask_field, None, iterations=40)

        # Bring back field to its original size.
        mask_field = cv2.dilate(mask_field, None, iterations=20)

        # Remove some noise at the edges.
        mask_field = cv2.erode(mask_field, None, iterations=3)

        # White mask
        lower_white = np.array(self._settings["white_mask"]["min"])
        upper_white = np.array(self._settings["white_mask"]["max"])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Intersect white mask and green mask. Thereby, only the parts of the
        # white mask that are inside the field are kept.
        return cv2.bitwise_and(mask_field, mask_white)

    @staticmethod
    def _remove_singular_points(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Remove points without any neighbors from the mask

        :param mask: The image mask.
        :return: The mask with all singular points removed.
        """
        for i in range(1, mask.shape[0] - 1):
            for j in range(1, mask.shape[1] - 1):
                if mask[i, j] == 1 and np.sum(mask[(i - 1) : (i + 1), (j - 1) : (j + 1)]) == 1:
                    mask[i, j] = 0
        return mask

    def _detect_lines(self, image: cv2.Mat) -> npt.NDArray[np.float_]:
        """Detect points on field lines in an image.

        :param image: The image. Ideally, in the area of the field no objects are present.
        :return: A list with the image coordinates of points on field lines.
        """
        mask_line = self._field_mask(image)
        skeleton_ = self._skeleton(mask_line)
        skeleton_ = self._remove_singular_points(skeleton_)
        points = np.array(np.where(skeleton_ > 0)).T.astype(float)
        return points[:, ::-1]

    @staticmethod
    def _correct_superview(
        points_in_center: npt.NDArray[np.float_], center: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Transforms points from a 16:9 image using superview to points in a 4:3 image.

        :param points_in_center: Superview pixel coordinates relative to the image center.
        :param center: The center of the original image.
        :return: Regular pixel coordinates relative to the image center.
        """
        o = np.array([5.0 / 4.0, 5.0 / 4.0])
        targets = np.array([1, 4.0 / 3.0])

        points_in_center_abs = np.abs(points_in_center)
        points_in_center_sign = np.sign(points_in_center)

        factors = o + points_in_center_abs / center * (np.subtract(targets, o))

        return points_in_center_abs * factors * points_in_center_sign

    @staticmethod
    def _uncorrect_superview(
        points_in_center: npt.NDArray[np.float_], center: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Transforms points from a 4:3 image to points in a 16:9 image using superview.

        :param points_in_center: Regular pixel coordinates relative to the image center.
        :param center: The center of the original image.
        :return: Superview pixel coordinates relative to the image center.
        """
        o = np.array([5.0 / 4.0, 5.0 / 4.0])
        targets = np.array([1, 4.0 / 3.0])

        points_in_center_abs = np.abs(points_in_center)
        points_in_center_sign = np.sign(points_in_center)

        p_2 = center * o / 2

        return (
            (np.sqrt(p_2**2 + points_in_center_abs * center * (np.subtract(targets, o))) - p_2)
            / (np.subtract(targets, o))
            * points_in_center_sign
        )

    @staticmethod
    def _correct_distortion(points_in_center: npt.NDArray[np.float_], intrinsics: Intrinsics) -> npt.NDArray[np.float_]:
        """Correct image distortion for a list of points.

        :param points_in_center: A list of pixel coordinates relative to the optical image center.
        :param intrinsics: The intrinsic parameters of the camera.
         :return: A list of pixel coordinates relative to the optical image center.
        """
        r2 = np.sum((points_in_center / intrinsics.focal_length) ** 2, axis=-1, keepdims=True)
        r4 = np.multiply(r2, r2)
        cr = 1 + intrinsics.k1 * r2 + intrinsics.k2 * r4 + intrinsics.k3 * r2 * r4
        return points_in_center * cr

    @staticmethod
    def _uncorrect_distortion(
        points_in_center: npt.NDArray[np.float_], intrinsics: Intrinsics
    ) -> npt.NDArray[np.float_]:
        """Distort the coordinates of a list of points.

        :param points_in_center: A list of undistorted pixel coordinates relative to the optical image center.
        :param intrinsics: The intrinsic parameters of the camera.
         :return: A list of distorted pixel coordinates relative to the optical image center.
        """
        r_primes = np.linalg.norm(points_in_center / intrinsics.focal_length, axis=-1)
        rs = []

        # This is the companion matrix of the 7th order polynomial k3*x^7+k2*x^5+k1*x^3+x.
        m = np.zeros((7, 7), dtype=np.float64)
        m.reshape(-1)[1::8] = 1
        m[1, 0] = -intrinsics.k2 / intrinsics.k3
        m[3, 0] = -intrinsics.k1 / intrinsics.k3
        m[5, 0] = -1 / intrinsics.k3
        for r_prime in r_primes:
            # Here we add the term -r_prime to the polynomial (or +r_prime as right-hand side of the equation).
            m[6, 0] = r_prime / intrinsics.k3
            roots = np.linalg.eigvals(m)

            # The positive real root is always the last one in the array.
            rs.append(np.real(roots[-1]))
        return (
            points_in_center
            / (np.linalg.norm(points_in_center, axis=-1, keepdims=True) + 1e-12)
            * intrinsics.focal_length
            * np.array(rs)[:, np.newaxis]
        )

    @staticmethod
    def _image2camera(points_in_image: npt.NDArray[np.float_], intrinsics: Intrinsics) -> npt.NDArray[np.float_]:
        """Transform points in image coordinates into camera-relative rays into the world.

        :param points_in_center: A list of points in image coordinates.
        :param intrinsics: The intrinsic parameters of the camera.
        :return: A list of camera-relative rays into the world.
        """
        assert len(points_in_image.shape) == 2
        assert points_in_image.shape[1] == 2

        if intrinsics.superview:
            center = intrinsics.resolution / 2
            points_in_center = points_in_image - center
            points_in_center = Camera._correct_superview(points_in_center, center)
            points_in_image = points_in_center + center

        points_in_center = points_in_image - intrinsics.optical_center
        points_in_center = Camera._correct_distortion(points_in_center, intrinsics)

        points_in_camera = np.zeros((points_in_center.shape[0], 3))
        points_in_camera[:, 0] = 1.0
        points_in_camera[:, 1] = -points_in_center[:, 0] / intrinsics.focal_length[0]
        points_in_camera[:, 2] = -points_in_center[:, 1] / intrinsics.focal_length[1]
        return points_in_camera

    @staticmethod
    def _camera2image(points_in_camera: npt.NDArray[np.float_], intrinsics: Intrinsics) -> npt.NDArray[np.float_]:
        """Transform camera-relative rays into the world into points in image coordinates.

        :param points_in_camera: A list of camera-relative rays.
        :param intrinsics: The intrinsic parameters of the camera.
        :return: A list of points in image coordinates.
        """
        assert len(points_in_camera.shape) == 2
        assert points_in_camera.shape[1] == 3

        # If things are at or behind the image plane, we have problems anyway...
        points_normalized = points_in_camera[:, 1:] / points_in_camera[:, :1]

        points_in_center = -points_normalized * intrinsics.focal_length

        points_in_center = Camera._uncorrect_distortion(points_in_center, intrinsics)
        points_in_image = points_in_center + intrinsics.optical_center

        if intrinsics.superview:
            center = intrinsics.resolution / 2
            points_in_center = points_in_image - center
            points_in_center = Camera._uncorrect_superview(points_in_center, center)
            points_in_image = points_in_center + center

        return points_in_image

    @staticmethod
    def _camera2world(
        points_in_camera: npt.NDArray[np.float_],
        pose: Extrinsics,
        z_in_world: float = 0.0,
    ) -> npt.NDArray[np.float_]:
        """Transform camera-relative rays into points in world coordinates.

        :param points_in_camera: Camera-relative rays into the world.
        :param pose: A pair of rotation matrix and camera position.
        :param z_in_world: The height of the horizontal plane above ground the points are projected to.
        :return: 2D points in world (i.e. field) coordinates.
        """

        # Ensure that the points are in the right format.
        assert len(points_in_camera.shape) == 2
        assert points_in_camera.shape[1] == 3

        # Rotate all points: cf. project()
        v = np.dot(pose.rotation, points_in_camera.transpose()).transpose()

        # Project all points: cf. project()
        # v[0:2]*(t[2]/(-v[2])) + t[0:2]
        return np.multiply(
            v[:, 0:2], np.tile(np.divide(z_in_world - pose.translation[2], v[:, 2]), (2, 1)).transpose()
        ) + np.tile(pose.translation[0:2], (v.shape[0], 1))

    @staticmethod
    def _world2camera(
        points_in_world: npt.NDArray[np.float_],
        pose: Extrinsics,
        z_in_world: float = 0,
    ) -> npt.NDArray[np.float_]:
        """Transform points in world coordinates into camera-relative rays.

        :param points_in_world: 2D points in world (i.e. field) coordinates.
        :param pose: A pair of rotation matrix and camera position.
        :param z_in_world: The z coordinate of all points in world coordinates.
        :return: Camera-relative rays into the world.
        """
        assert len(points_in_world.shape) == 2
        assert points_in_world.shape[1] == 2

        return np.dot(
            np.concatenate((points_in_world, np.full((points_in_world.shape[0], 1), z_in_world)), axis=1), pose.rotation
        ) - np.dot(np.transpose(pose.rotation), pose.translation)

    @staticmethod
    def _make_field_points(field: Field, step: float) -> npt.NDArray[np.float_]:
        """Create a list of points from the field lines in the field model.

        :param field: The field model.
        :param step: The distance between neighboring points (in m).
        :return: The list of points on field lines.
        """
        points: list[list[float]] = []

        length_half = field.field_length / 2.0
        width_half = field.field_width / 2.0
        penalty_area_width_half = field.penalty_area_width / 2.0

        # side lines
        for x in np.arange(-length_half, length_half + step, step):
            points.append([x, -width_half])
            points.append([x, width_half])

        # goal lines and the center line
        for y in np.arange(-width_half, width_half + step, step):
            points.append([-length_half, y])
            points.append([length_half, y])
            points.append([0.0, y])

        # penalty area long lines
        for y in np.arange(-penalty_area_width_half, penalty_area_width_half + step, step):
            points.append([-length_half + field.penalty_area_length, y])
            points.append([length_half - field.penalty_area_length, y])

        # penalty area short lines
        for x in np.arange(0.0, field.penalty_area_length, step):
            points.append([-length_half + x, -penalty_area_width_half])
            points.append([-length_half + x, penalty_area_width_half])
            points.append([length_half - x, -penalty_area_width_half])
            points.append([length_half - x, penalty_area_width_half])

        if field.has_goal_area:
            goal_area_width_half = field.goal_area_width / 2.0

            # goal area long lines
            for y in np.arange(-goal_area_width_half, goal_area_width_half + step, step):
                points.append([-length_half + field.goal_area_length, y])
                points.append([length_half - field.goal_area_length, y])

            # goal area short lines
            for x in np.arange(0.0, field.goal_area_length, step):
                points.append([-length_half + x, -goal_area_width_half])
                points.append([-length_half + x, goal_area_width_half])
                points.append([length_half - x, -goal_area_width_half])
                points.append([length_half - x, goal_area_width_half])

        # penalty mark
        # points.append([length_half-f.penalty_cross_distance, 0.0])
        # points.append([-length_half+f.penalty_cross_distance, 0.0])

        # middle circle
        number_of_steps = (np.pi * field.center_circle_diameter) / step
        for a in np.arange(-np.pi, np.pi, 2.0 * np.pi / number_of_steps):
            points.append(
                [field.center_circle_diameter * np.sin(a) * 0.5, field.center_circle_diameter * np.cos(a) * 0.5]
            )

        return np.array(points).astype(float)

    @staticmethod
    def _find_closest_points(model: KDTree, data: npt.NDArray[np.float_]) -> tuple[list[int], float]:
        """Find the closest point in the model for each data point.

        A kd-tree is built from the model to speed up searching.
        :param model: A kd tree of points on field lines created from the model.
        :param data: A list of points on field lines found in the image.
        :return: A list of indexes of the closest model points and the average distance (in m).
        """
        result: list[int] = []
        deviation = 0.0

        if data.shape[0] == 0:
            return result, deviation

        ordered_errors, ordered_neighbors = model.query(data, k=1)
        for idx, e in zip(ordered_neighbors, ordered_errors):
            result.append(idx)
            deviation += e * e

        return result, deviation / float(data.shape[0])

    @staticmethod
    def _filter_outliers(
        model: npt.NDArray[np.float_], model_as_tree: KDTree, data: npt.NDArray[np.float_], max_error: float
    ) -> list[int]:
        """
        Filter outliers from the measured points on field lines that are too far away from model points.

        :param model: A list of points on field lines created from the model.
        :param model_as_tree: The model points in a kd tree.
        :param data: A list of points on field lines found in the image.
        :param max_error: The maximum distance a measured point can deviate from the model (in m).
        :return: A list of indexes into `data` of points that are close enough to the model.
        """
        normal_indexes: list[int] = []

        c_idx, _ = Camera._find_closest_points(model_as_tree, data)
        for i_data, i_model in enumerate(c_idx):
            if i_data < data.shape[0] and i_model < model.shape[0]:
                if np.linalg.norm(data[i_data, :] - model[i_model, :]) <= max_error:
                    normal_indexes.append(i_data)

        return normal_indexes

    @staticmethod
    def _error_mean_square(
        model: npt.NDArray[np.float_],
        points: npt.NDArray[np.float_],
        t: npt.NDArray[np.float_],
        static_intrinsics: npt.NDArray[np.float_],
    ) -> float:
        """Determines the mean squared distance measured points deviate from model points.

        :param model: A list of closest model points to the points in `points`
        :param points: A list of camera-relative rays targeting field lines found in the image.
        :param t: The tested camera calibration without static intrinsic parameters.
        :param static_intrinsics: The superview flag and the image resolution.
        :return: The mean squared distance measured points deviate from model points (in m^2).
        """
        t = np.concatenate([t, static_intrinsics])
        points_in_camera = Camera._image2camera(points, Intrinsics.from_vector(t[6:]))
        t_points = Camera._camera2world(points_in_camera, Extrinsics.from_vector(t[:6]))

        # calculate error
        assert model.shape == t_points.shape
        e = model - t_points
        e = np.multiply(e, e)

        return np.sum(np.sum(e, axis=0)) / float(points.shape[0])

    @staticmethod
    def _find_transformation(
        model: npt.NDArray[np.float_],
        model_as_tree: KDTree,
        points: npt.NDArray[np.float_],
        t0: npt.NDArray[np.float_],
        iterations: int,
    ) -> tuple[npt.NDArray[np.float_], float]:
        """Find the extrinsic camera calibration.

        :param model: A list of points on field lines created from the model.
        :param model_as_tree: The model points in a kd tree.
        :param points: A list of camera-relative rays targeting field lines found in the image.
        :param t0: An initial guess of the calibration.
        :param iterations: The maximum number of iterations.
        :return: The extrinsic and intrinsic camera calibration. In addition, the registration error is returned.
        """
        t = t0
        result = None
        bounds = (
            (-360, 360),  # rotation around x axis
            (-360, 360),  # rotation around y axis
            (-360, 360),  # rotation around z axis
            (None, None),  # x position
            (None, None),  # y position
            (None, None),  # z position
            (0, t[-2]),  # horizontal optical center (limited by image width)
            (0, t[-1]),  # vertical optical center (limited by image height)
            (None, None),  # horizontal focal length
            (None, None),  # vertical focal length
            (None, None),  # k1
            (None, None),  # k2
            (None, None),  # k3
        )

        for k in range(iterations):
            logger.debug("Iteration %s", k)

            # make the assignment only once
            points_in_camera = Camera._image2camera(points, Intrinsics.from_vector(t[6:]))
            t_points = Camera._camera2world(points_in_camera, Extrinsics.from_vector(t[:6]))
            c_idx, _ = Camera._find_closest_points(model_as_tree, t_points)
            model_selection = model[c_idx]

            # Solve the problem, but don't optimize superview and resolution (i.e. the last 3 elements)
            result = optimize.minimize(
                lambda x: Camera._error_mean_square(model_selection, points, x, t[-3:]),
                t[:-3],
                method="SLSQP",
                bounds=bounds,
            )
            t = np.concatenate([result.x, t[-3:]])
            logger.debug("Error: %sm", np.sqrt(result.fun))

        if result is not None and result.success:
            logger.debug("Success: %s", str(t))
            return t, np.sqrt(result.fun)

        logger.error(result)
        raise RuntimeError("[ERROR] could not solve")

    def _align_camera(
        self,
        source: str,
        field: Field,
        points: npt.NDArray[np.float_],
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Find and set the intrinsic and extrinsic camera calibration.

        :param source: Path of the video.
        :param field: The field model.
        :param points: A list of points on field lines found in the image.
        :return: The image points that are close enough to model points and the image points transformed to world
        coordinates using the camera pose found.
        """

        # the model for the field lines
        model = self._make_field_points(field, step=0.02)
        model_as_tree = KDTree(model)

        # initial guess for camera calibration
        t0 = np.array(
            self._get_initial_rotation_guess(source)
            + self._settings["calibration"]["initial_extrinsics"]["translation"]
            + self._settings["calibration"]["initial_intrinsics"]["optical_center"]
            + self._settings["calibration"]["initial_intrinsics"]["focal_length"]
            + [
                self._settings["calibration"]["initial_intrinsics"]["k1"],
                self._settings["calibration"]["initial_intrinsics"]["k2"],
                self._settings["calibration"]["initial_intrinsics"]["k3"],
                1.0 if self._settings["calibration"]["initial_intrinsics"]["superview"] else 0.0,
            ]
            + self._settings["calibration"]["initial_intrinsics"]["resolution"],
        )

        # project the points with the initial transform t0
        points_in_camera = self._image2camera(points, Intrinsics.from_vector(t0[6:]))
        t_points = self._camera2world(points_in_camera, Extrinsics.from_vector(t0))

        # ignore the 'worst outliers', i.e. those more than a threshold away from the model
        normal_indexes = self._filter_outliers(
            model, model_as_tree, t_points, self._settings["calibration"]["outlier_threshold"]
        )
        points = points[normal_indexes]

        # optimize the alignment
        t, _ = self._find_transformation(model, model_as_tree, points, t0, self._settings["calibration"]["iterations"])

        # project the points with the final transformation
        self.intrinsics = Intrinsics.from_vector(t[6:])
        self._extrinsics = Extrinsics.from_vector(t[:6])
        assert self.intrinsics is not None
        assert self._extrinsics is not None
        points_in_camera = self._image2camera(points, self.intrinsics)
        return points, self._camera2world(points_in_camera, self._extrinsics), t_points

    def _get_initial_rotation_guess(self, source: str | Path) -> list[float]:
        """Get an initial guess for the rotation of the camera, either directly from configuration or augmented by GPMF
        telemetry data embedded into the video file.

        :param source: Path of the video.
        :return: An initial guess for the rotation of the camera as list of xyz-Euler angles in degrees.
        """
        if isinstance(source, str):
            source = Path(source)
        try:
            n = 10
            if source.suffix == ".txt":
                with source.open() as f:
                    video: str = f.readline()[:-1]
            else:
                video = str(source)
            telemetry = GPMFParser(video).telemetry()
            gravity_vector = list(
                itertools.islice(
                    (
                        [item["x"], item["y"], item["z"]]
                        for data in telemetry
                        if "GravityVector" in data
                        for item in data["GravityVector"]["Data"]
                    ),
                    n,
                )
            )
            if len(gravity_vector) == n:
                grav = np.array(gravity_vector, dtype=np.float32)[-1]
                gopro2camera = R.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
                camera2world_guessed = R.from_euler(
                    "z", self._settings["calibration"]["initial_extrinsics"]["rotation"][2], degrees=True
                )

                reference_grav = [0, 1, 0]
                orth_a = np.cross(grav, reference_grav)
                angle = np.arccos((grav @ reference_grav) / np.linalg.norm(grav))
                gp_rot_measured = R.from_rotvec(orth_a / np.linalg.norm(orth_a) * angle)

                r = camera2world_guessed * gopro2camera * gp_rot_measured * gopro2camera.inv()

                return r.as_euler("xyz", degrees=True).tolist()
        except OSError:
            # video file does not contain a metadata stream
            pass

        return self._settings["calibration"]["initial_extrinsics"]["rotation"]
