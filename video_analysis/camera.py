"""This module uses a lot of the code from the repository https://github.com/BerlinUnited/RoboCupTools
(some of it modified though).
It contains everything needed to calculate an aerial view of the game that has been recorded from a side view.
"""
from __future__ import annotations

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

    def scale(self, resolution: npt.NDArray[np.float_]) -> None:
        """Scales intrinsics to be used with another resolution.

        This only works if the active region of the camera (and all the lens settings)
        are the same for both resolutions.
        :param resolution: The new resolution
        """
        ratio = resolution / self.resolution
        self.optical_center *= ratio
        self.focal_length *= ratio
        self.resolution = resolution


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


def read_video_background(dataset: SourceAdapter, images: int) -> cv2.Mat:
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
    return cv2.cvtColor(subtractor.getBackgroundImage(), cv2.COLOR_BGR2RGB)


def skeleton(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
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


def field_mask(image: cv2.Mat, settings: dict[str, Any]) -> npt.NDArray[np.uint8]:
    """Generate mask from the image that only contains the field lines.

    :param image: The image.
    :param settings: The settings containing the color space bounds to segment the field
    and the field lines.
    """

    # Convert image to the HSV color space.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green mask
    lower_green = np.array(settings["green_mask"]["min"])
    upper_green = np.array(settings["green_mask"]["max"])
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
    lower_white = np.array(settings["white_mask"]["min"])
    upper_white = np.array(settings["white_mask"]["max"])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Intersect white mask and green mask. Thereby, only the parts of the
    # white mask that are inside the field are kept.
    return cv2.bitwise_and(mask_field, mask_white)


def remove_singular_points(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Remove points without any neighbors from the mask

    :param mask: The image mask.
    :return: The mask with all singular points removed.
    """
    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if mask[i, j] == 1 and np.sum(mask[(i - 1) : (i + 1), (j - 1) : (j + 1)]) == 1:
                mask[i, j] = 0
    return mask


def mask_to_point_list(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.float_]:
    """Convert a mask to a list of points.

    :param mask: The mask.
    :return: A list of the coordinates of all masked points.
    """
    return np.array(np.where(img > 0)).T.astype(float)


def detect_lines(image: cv2.Mat, settings: dict[str, Any]) -> npt.NDArray[np.float_]:
    """Detect points on field lines in an image.

    :param image: The image. Ideally, in the area of the field no objects are present.
    :param settings: The settings containing the color space bounds to segment the field
    and the field lines.
    :return: A list with the image coordinates of points on field lines.
    """
    mask_line = field_mask(image, settings)
    skeleton_ = skeleton(mask_line)
    skeleton_ = remove_singular_points(skeleton_)
    points = mask_to_point_list(skeleton_)
    return points[:, ::-1]


def correct_superview(
    points_in_center: npt.NDArray[np.float_], center: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """Transforms points from a 16:9 image using superview to points in a 4:3 image.

    :param points_in_center: Superview pixel coordinates relative to the image center.
    :param optical_center: The center of the original image.
    :return: Regular pixel coordinates relative to the image center.
    """
    o = np.array([5.0 / 4.0, 5.0 / 4.0])
    targets = np.array([1, 4.0 / 3.0])

    points_in_center_abs = np.abs(points_in_center)
    points_in_center_sign = np.sign(points_in_center)

    factors = o + points_in_center_abs / center * (np.subtract(targets, o))

    return points_in_center_abs * factors * points_in_center_sign


def uncorrect_superview(
    points_in_center: npt.NDArray[np.float_], center: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """Transforms points from a 4:3 image to points in a 16:9 image using superview.

    :param points_in_center: Regular pixel coordinates relative to the image center.
    :param optical_center: The center of the original image.
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


def correct_distortion(points_in_center: npt.NDArray[np.float_], intrinsics: Intrinsics) -> npt.NDArray[np.float_]:
    """Correct image distortion for a list of points.

    :param points_in_center: A list of pixel coordinates relative to the optical image center.
    :param intrinsics: The intrinsic parameters of the camera.
    :return: A list of pixel coordinates relative to the optical image center.
    """
    r2 = np.sum((points_in_center / intrinsics.focal_length) ** 2, axis=-1, keepdims=True)
    r4 = np.multiply(r2, r2)
    cr = 1 + intrinsics.k1 * r2 + intrinsics.k2 * r4 + intrinsics.k3 * r2 * r4
    return points_in_center * cr


def uncorrect_distortion(points_in_center: npt.NDArray[np.float_], intrinsics: Intrinsics) -> npt.NDArray[np.float_]:
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


def image2camera(points_in_image: npt.NDArray[np.float_], intrinsics: Intrinsics) -> npt.NDArray[np.float_]:
    """Transform points in image coordinates into camera-relative rays into the world.

    :param points_in_center: A list of points in image coordinates.
    :param intrinsics: The intrinsic parameters of the camera.
    :return: A list of camera-relative rays into the world.
    """
    assert len(points_in_image.shape) == 2
    assert points_in_image.shape[1] == 2

    points_in_center = points_in_image - intrinsics.optical_center

    if intrinsics.superview:
        # TODO: Probably use the coordinate center instead of the optical center here.
        points_in_center = correct_superview(points_in_center, intrinsics.optical_center)

    points_in_center = correct_distortion(points_in_center, intrinsics)

    points_in_camera = np.zeros((points_in_center.shape[0], 3))
    points_in_camera[:, 0] = 1.0
    points_in_camera[:, 1] = -points_in_center[:, 0] / 820.0  # TODO: intrinsics.focal_length[0]
    points_in_camera[:, 2] = -points_in_center[:, 1] / 820.0  # TODO: intrinsics.focal_length[1]
    return points_in_camera


def camera2image(points_in_camera: npt.NDArray[np.float_], intrinsics: Intrinsics) -> npt.NDArray[np.float_]:
    """Transform camera-relative rays into the world into points in image coordinates.

    :param points_in_camera: A list of camera-relative rays.
    :param intrinsics: The intrinsic parameters of the camera.
    :return: A list of points in image coordinates.
    """
    assert len(points_in_camera.shape) == 2
    assert points_in_camera.shape[1] == 3

    # If things are at or behind the image plane, we have problems anyway...
    points_normalized = points_in_camera[:, 1:] / points_in_camera[:, :1]

    points_in_center = -points_normalized * 820.0  # TODO: intrinsics.focal_length

    points_in_center = uncorrect_distortion(points_in_center, intrinsics)

    if intrinsics.superview:
        # TODO: Probably use the coordinate center instead of the optical center here.
        points_in_center = uncorrect_superview(points_in_center, intrinsics.optical_center)

    return points_in_center + intrinsics.optical_center


def camera2world(
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


def world2camera(
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


def make_field_points(field: Field, step: float = 0.2) -> npt.NDArray[np.float_]:
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
        points += [[x, -width_half]]
        points += [[x, width_half]]

    # goal lines and the center line
    for y in np.arange(-width_half, width_half + step, step):
        points += [[-length_half, y]]
        points += [[length_half, y]]
        points += [[0.0, y]]

    # penalty area long lines
    for y in np.arange(-penalty_area_width_half, penalty_area_width_half + step, step):
        points += [[-length_half + field.penalty_area_length, y]]
        points += [[length_half - field.penalty_area_length, y]]

    # penalty area short lines
    for x in np.arange(0.0, field.penalty_area_length, step):
        points += [[-length_half + x, -penalty_area_width_half]]
        points += [[-length_half + x, penalty_area_width_half]]
        points += [[length_half - x, -penalty_area_width_half]]
        points += [[length_half - x, penalty_area_width_half]]

    if field.has_goal_area:
        goal_area_width_half = field.goal_area_width / 2.0

        # goal area long lines
        for y in np.arange(-goal_area_width_half, goal_area_width_half + step, step):
            points += [[-length_half + field.goal_area_length, y]]
            points += [[length_half - field.goal_area_length, y]]

        # goal area short lines
        for x in np.arange(0.0, field.goal_area_length, step):
            points += [[-length_half + x, -goal_area_width_half]]
            points += [[-length_half + x, goal_area_width_half]]
            points += [[length_half - x, -goal_area_width_half]]
            points += [[length_half - x, goal_area_width_half]]

    # penalty mark
    # points += [[length_half-f.penalty_cross_distance, 0.0]]
    # points += [[-length_half+f.penalty_cross_distance, 0.0]]

    # middle circle
    number_of_steps = (np.pi * field.center_circle_diameter) / step
    for a in np.arange(-np.pi, np.pi, 2.0 * np.pi / number_of_steps):
        points += [[field.center_circle_diameter * np.sin(a) * 0.5, field.center_circle_diameter * np.cos(a) * 0.5]]

    return np.array(points).astype(float)


def find_closest_points(model: npt.NDArray[np.float_], data: npt.NDArray[np.float_]) -> tuple[list[int], float]:
    """Find the closest point in the model for each data point.

    A kd-tree is built from the model to speed up searching.
    :param model: A list of points on field lines created from the model.
    :param data: A list of points on field lines found in the image.
    :return: A list of indexes of the closest model points and the average distance (in m).
    """
    result: list[int] = []
    deviation = 0.0

    if data.shape[0] == 0:
        return result, deviation

    # TODO: This kd tree is always the same, since the model is constant. Only compute it once.
    tree = KDTree(model)
    ordered_errors, ordered_neighbors = tree.query(data, k=1)
    for idx, e in zip(ordered_neighbors, ordered_errors):
        result += [idx]
        deviation += e * e

    return result, deviation / float(data.shape[0])


def filter_outliers(model: npt.NDArray[np.float_], data: npt.NDArray[np.float_], max_error: float) -> list[int]:
    """
    Filter outliers from the measured points on field lines that are too far away from model points.

    :param model: A list of points on field lines created from the model.
    :param data: A list of points on field lines found in the image.
    :param max_error: The maximum distance a measured point can deviate from the model (in m).
    :return: A list of indexes into `data` of points that are close enough to the model.
    """
    normal_indexes: list[int] = []

    c_idx, _ = find_closest_points(model, data)
    for i_data, i_model in enumerate(c_idx):
        if i_data < data.shape[0] and i_model < model.shape[0]:
            v = data[i_data, :] - model[i_model, :]
            if np.hypot(v[0], v[1]) <= max_error:
                normal_indexes += [i_data]

    return normal_indexes


def error(model: npt.NDArray[np.float_], points: npt.NDArray[np.float_], t: npt.NDArray[np.float_]) -> float:
    """Determines the average distance measured points deviate from model points.

    :param model: A list of points on field lines created from the model.
    :param points: A list of camera-relative rays targeting field lines found in the image.
    :param t: The pose as rotations around three axes (in degrees) and the position (in m).
    :return: The average distance measured points deviate from model points (in m).
    """
    points = camera2world(points, Extrinsics.from_vector(t))
    _, e = find_closest_points(model, points)

    logger.debug("Error: %sm", np.sqrt(e))
    return e


def error_mean_square(
    model: npt.NDArray[np.float_], points: npt.NDArray[np.float_], t: npt.NDArray[np.float_]
) -> float:
    """Determines the mean squared distance measured points deviate from model points.

    :param model: A list of closest model points to the points in `points`
    :param points: A list of camera-relative rays targeting field lines found in the image.
    :param t: The pose as rotations around three axes (in degrees) and the position (in m).
    :return: The mean squared distance measured points deviate from model points (in m^2).
    """
    t_points = camera2world(points, Extrinsics.from_vector(t))
    assert model.shape == t_points.shape

    # calculate error
    e = model - t_points
    e = np.multiply(e, e)

    return np.sum(np.sum(e, axis=0)) / float(points.shape[0])


def registration_fast(
    model: npt.NDArray[np.float_],
    points: npt.NDArray[np.float_],
    t0: npt.NDArray[np.float_],
    iterations: int = 10,
) -> tuple[npt.NDArray[np.float_], float]:
    """Find the extrinsic camera calibration through optimization.

    :param model: A list of points on field lines created from the model.
    :param points: A list of camera-relative rays targeting field lines found in the image.
    :param t0: An initial guess of the pose as rotations around three axes (in degrees) and the position (in m).
    :param iterations: How many optimization iterations should be performed?
    :return: The optimal camera pose as rotations around three axes (in degrees) and the position (in m).
    """
    t = t0
    result = None

    for k in range(iterations):
        logger.debug("Iteration %s", k)

        # make the assignment only once
        t_points = camera2world(points, Extrinsics.from_vector(t))
        c_idx, _ = find_closest_points(model, t_points)
        model_selection = model[c_idx, :]

        # Solve the problem
        def function(x):
            # pylint: disable-next=cell-var-from-loop
            return error_mean_square(model_selection, points, x)

        bounds = ((-360, 360), (-360, 360), (-360, 360), (None, None), (None, None), (None, None))
        result = optimize.minimize(function, t, method="SLSQP", bounds=bounds)

        t = result.x
        e = np.sqrt(result.fun)
        logger.debug("Error: %sm", e)

    if result is not None and result.success:
        logger.debug("Success: %s", str(result.x))
        return result.x, np.sqrt(result.fun)
    logger.error(result)
    raise RuntimeError("[ERROR] could not solve")


def find_transformation(
    model_points: npt.NDArray[np.float_],
    points: npt.NDArray[np.float_],
    t0: npt.NDArray[np.float_],
    iterations: int,
    registration_function=registration_fast,
) -> tuple[npt.NDArray[np.float_], float]:
    """Find the extrinsic camera calibration.

    :param model: A list of points on field lines created from the model.
    :param points: A list of camera-relative rays targeting field lines found in the image.
    :param t0: An initial guess of the pose as rotations around three axes (in degrees) and the position (in m).
    :param iterations: The maximum number of iterations.
    :param registration_function: The registration function to use.
    :return: The camera pose as rotations around three axes (in degrees) and the position (in m). In addition,
    the registration error is returned.
    """
    # Calculate the initial error
    e_initial = np.sqrt(error(model_points, points, t0))
    logger.debug("Initial Error: %sm", e_initial)

    t, err = registration_function(model_points, points, t0, iterations)

    return t, err


# This function takes an array of points from one perspective and returns all the equivalent points from the second
# perspective
def align_camera(
    field: Field,
    points: npt.NDArray[np.float_],
    intrinsics: Intrinsics,
    settings: dict[str, Any],
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], Extrinsics]:
    """Find the extrinsic camera calibration.

    :param field: The field model.
    :param points: A list of points on field lines found in the image.
    :param intrinsics: The intrinsic camera calibration.
    :param settings: The settings that contain the initial guess of the camera's pose as well as the color
    space bounds for the field and the field lines.
    :return: The image points that are close enough to model points, the image points transformed to world
    coordinates using the camera pose found, and the camera pose itself as rotations around three axes
    (in degrees) and the position (in m).
    """
    points_original = points
    points = image2camera(points, intrinsics)

    # the model for the RC19 field lines
    model_points = make_field_points(field, step=0.05)

    # initial guess for the position of the camera at the RC19
    t0 = np.array(
        settings["calibration"]["initial_camera_pose"]["rotation"]
        + settings["calibration"]["initial_camera_pose"]["translation"]
    )

    # project the points with the initial transform t0
    t_points = camera2world(points, Extrinsics.from_vector(t0))

    # ignore the 'worst outliers', i.e., more than 3m away
    normal_indexes = filter_outliers(model_points, t_points, 3)
    points_reduced = points_original[normal_indexes, :]
    points = points[normal_indexes, :]

    # optimize the alignment
    t, _ = find_transformation(
        make_field_points(field, step=0.2), points, t0, settings["calibration"]["iterations"], registration_fast
    )

    # project the points with the final transformation
    pose = Extrinsics.from_vector(t)
    final_points = camera2world(points, pose)
    return points_reduced, final_points, pose


class Camera:
    """Provides transformations between camera (i.e. image) coordinates and world (i.e. field)
    coordinates.

    The origin of the world coordinates is at the center point of the field. The first dimension
    points to the right. The second dimension points away from the camera. The third dimension
    points upward. World coordinates are in meters.
    This class also stores the number of frames per second at which the video is played back.
    """

    def __init__(self, fps: float) -> None:
        """Initialize the mapper between image and world coordinates.

        :param fps: The number of frames per second at which the video is played back.
        """
        self.fps: float = fps
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
        settings: dict[str, Any],
        calibration_path: Path,
        force: bool = False,
        skip: bool = False,
        verbose: bool = False,
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
        """
        if not force and calibration_path.is_file():
            with calibration_path.open(encoding="UTF-8", newline="\n") as file:
                calibration: dict[str, Any] = json.load(file)
                self._extrinsics = Extrinsics.from_dict(calibration["extrinsics"])
                self.intrinsics = Intrinsics.from_dict(calibration["intrinsics"])
            return

        with (ROOT / "config" / "GoProHERO5.json").open(encoding="UTF-8", newline="\n") as file:
            intrinsics: dict[str, Any] = json.load(file)
            self.intrinsics = Intrinsics.from_dict(intrinsics)
        assert self.intrinsics is not None

        if skip:
            logger.error("Skipping camera calibration. Using default values.")
            self._extrinsics = Extrinsics.from_vector(
                settings["calibration"]["initial_camera_pose"]["rotation"]
                + settings["calibration"]["initial_camera_pose"]["translation"]
            )
            return

        log_level = logging.INFO
        if verbose:
            # Change logging level to print all debug messages during calibration.
            log_level = logger.level
            logger.setLevel(logging.DEBUG)
        logger.info("Calibrating camera...")
        # Load data once
        dataset = SourceAdapter(
            source, imgsz, stride, pt, step=settings["calibration"]["step_size"]
        )  # We only select every hundredth image for a better quality of the calibration

        background = read_video_background(dataset, settings["calibration"]["images"])
        self.intrinsics.scale(np.array(np.shape(background)[1::-1], dtype=np.float32))
        points_original = detect_lines(background, settings)
        try:
            points_reduced, points_transformed, self._extrinsics = align_camera(
                field, points_original, self.intrinsics, settings
            )
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
                ax1.plot(points_original[:, 0], background.shape[0] - points_original[:, 1], ".")
                ax1.set_title("Detected line points")
                ax2.set_aspect("equal")
                ax2.plot(points_reduced[:, 0], background.shape[0] - points_reduced[:, 1], ".")
                ax2.set_title("reduced line points")
                plt.savefig(path / "before.png")
                _, (ax3) = plt.subplots(1, 1)
                ax3.set_aspect("equal")
                ax3.plot(points_transformed[:, 0], points_transformed[:, 1], ".")
                ax3.set_title("final projection")
                plt.savefig(path / "after.png")
                logger.setLevel(log_level)
            logger.info("Calibration finished.")

        except RuntimeError:
            logger.error("Calibration failed. Using default values.")
            self._extrinsics = Extrinsics.from_vector(
                settings["calibration"]["initial_camera_pose"]["rotation"]
                + settings["calibration"]["initial_camera_pose"]["translation"]
            )

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
        points_in_camera = image2camera(points_in_image, self.intrinsics)
        points_in_world = camera2world(points_in_camera, self._extrinsics, z_in_world=z)
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
        points_in_camera = world2camera(points_in_world, self._extrinsics, z_in_world=z)
        points_in_image = camera2image(points_in_camera, self.intrinsics)
        return points_in_image[0] if single else points_in_image
