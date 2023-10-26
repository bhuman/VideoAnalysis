from __future__ import annotations

from math import ceil

import cairo
import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module
from scipy.spatial._qhull import QhullError
from shapely.geometry import Polygon

from ..world_model import WorldModel
from ..world_model.players import Player
from .localization import Localization


class ControlledAreas:
    """This class implements a provider for statistics about the average size of the controlled
    area per team.

    A position on the field is assumed to be controlled by a team if it is closer to a player
    of that team than to any player of the other team. The sum of all these positions is the
    area controlled by the team. These areas are computed by constructing the Voronoi diagram
    from the positions of all players on the field.
    """

    _max_line_length = 2
    """The maximum length of a single line drawn. Longer lines are split."""

    def __init__(self, world_model: WorldModel, categories: dict[str, list]) -> None:
        """Initialize the provider for the average controlled areas per team.

        :param world_model: The world model.
        :param categories: The statistics category provided by this provider is added to the
        dictionary of all categories.
        """
        self._world_model: WorldModel = world_model
        self._categories: dict[str, list] = categories
        self._categories.update({"Controlled Area": [0, 0, " m²"]})
        self._vertices: npt.NDArray[np.float_] | None = None
        self._regions: list[list[int]] | None = None
        self._field_boundary = Polygon(
            [
                [-self._world_model.field.field_length / 2, -self._world_model.field.field_width / 2],
                [-self._world_model.field.field_length / 2, self._world_model.field.field_width / 2],
                [self._world_model.field.field_length / 2, self._world_model.field.field_width / 2],
                [self._world_model.field.field_length / 2, -self._world_model.field.field_width / 2],
            ]
        )
        self._area_per_second_sums: list[float] = [0, 0]
        self._time: float = 0

        self._positions: list[np.ndarray] = []
        self._colors: list[Player.Color] = []
        self._color_triples: list[tuple[float, ...]] = []
        self.localization: Localization | None = None

    def update(self) -> None:
        """Update the controlled areas category."""
        self._positions = []
        self._colors = []
        self._color_triples = []
        players: list[Player] = self._world_model.players.players
        if self.localization is not None:
            players = self.localization.filter_players(players)
        for player in players:  # Iterate through players
            self._positions.append(player.position)
            self._colors.append(player.color)
            self._color_triples.append(player.color.as_color_triple(range01=True))

        self._regions = None
        if self._world_model.game_state.state == self._world_model.game_state.State.PLAYING:
            regions, vertices = self._calc_regions()
            for i, region in enumerate(regions):
                poly = Polygon([vertices[v] for v in region])
                poly = Polygon(poly.intersection(self._field_boundary))
                assert poly.exterior is not None
                poly = Polygon(poly.exterior.coords)
                self._area_per_second_sums[self._world_model.players.color_to_team(self._colors[i])] += (
                    poly.area / self._world_model.camera.fps
                )

            self._time += 1 / self._world_model.camera.fps
            for team in range(2):
                self._categories["Controlled Area"][team] = round(self._area_per_second_sums[team] / self._time, 1)

    def _calc_regions(self) -> tuple[list[list[int]], npt.NDArray[np.float_]]:
        """Compute the Voronoi regions and returns them.

        Note that `self._regions` is used to cache these regions to avoid computing them
        multiple times for the same frame.
        :return: The regions (indexes into the vertices) and the vertices.
        """

        # If the regions were already computed, return them.
        if self._regions is not None:
            assert self._vertices is not None
            return self._regions, self._vertices

        # FIXME the following should prevent a crash
        # maybe there is a better solution such as returning the last regions
        if len(self._positions) == 0:
            return [], np.array([], dtype=np.float_)
        # Compute Voronoi diagram. It can fail (e.g. too few points, all on a line, etc.).
        try:
            vor = Voronoi(self._positions)
        except QhullError:
            self._regions = []
            self._vertices = np.asarray([])
            return self._regions, self._vertices

        new_regions = []
        new_vertices = vor.vertices.tolist()
        radius = vor.points.ptp().max() * 2
        center = vor.points.mean(axis=0)

        # Construct a map containing all ridges for a given point
        all_ridges: dict[int, list[tuple[int, int, int]]] = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            if p1 not in all_ridges:
                continue

            ridges: list[tuple[int, int, int]] = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge
                tangent = vor.points[p2] - vor.points[p1]
                tangent /= np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, normal)) * normal
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region_np = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region_np.tolist())

        self._regions = new_regions
        self._vertices = np.asarray(new_vertices)
        return self._regions, self._vertices

    def draw_on_field(self, context: cairo.Context) -> None:
        """Draw the controlled areas onto the field.

        :param context: The context to draw to.
        """
        regions, vertices = self._calc_regions()
        for i, region in enumerate(regions):
            poly = Polygon(vertices[region])
            poly = Polygon(poly.intersection(self._field_boundary))
            assert poly.exterior is not None
            polygon = list(poly.exterior.coords)
            if len(polygon) > 0:
                context.move_to(polygon[-1][0], polygon[-1][1])
                for v in polygon:
                    context.line_to(v[0], v[1])

                color = self._color_triples[i]
                context.set_source_rgba(color[0], color[1] / 5, color[2], 0.2)
                context.fill_preserve()

                context.set_source_rgb(0, 0, 0)
                context.set_line_width(0.04)
                context.stroke()

    def draw_on_image(self, image: npt.NDArray[np.uint8]) -> None:
        """Draw the controlled areas onto the image.

        :param image: The image to draw onto.
        """
        regions, vertices = self._calc_regions()
        for region in regions:
            poly = Polygon(vertices[region])
            poly = Polygon(poly.intersection(self._field_boundary))
            assert poly.exterior is not None
            poly = np.asarray(poly.exterior.coords)
            if len(poly) > 0:
                polygon = self._world_model.camera.world2image(poly)
                cv2.polylines(image, [self._split(polygon).astype(np.int32)], True, (0, 192, 0), 2)

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
