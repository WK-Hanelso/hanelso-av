from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    PolygonMapObject,
    PolylineMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import RasterLayer, RasterMap, SemanticMapLayer

_EPS = 1e-6
_LANE_INDEX_SPACING_M = 1.0
_BASELINE_SPACING_M = 0.5


def _as_xy_array(points: Sequence[Sequence[float]]) -> npt.NDArray[np.float64]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected NxM points with M>=2, got shape={arr.shape}")
    return arr[:, :2]


def _point_xy(point: Point2D | Point | Sequence[float]) -> Tuple[float, float]:
    if isinstance(point, Point):
        return float(point.x), float(point.y)
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    if isinstance(point, Sequence) and len(point) >= 2:
        return float(point[0]), float(point[1])
    raise TypeError(f"Unsupported point type: {type(point)!r}")


def _shapely_point(point: Point2D | Point | Sequence[float]) -> Point:
    x, y = _point_xy(point)
    return Point(x, y)


def _cumulative_lengths(xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(xy) == 1:
        return np.zeros((1,), dtype=np.float64)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg)])


def _sample_polyline_at_progress(
    xy: npt.NDArray[np.float64],
    cumulative: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if len(xy) == 0:
        return np.zeros((len(targets), 2), dtype=np.float64)
    if len(xy) == 1:
        return np.repeat(xy[:1], len(targets), axis=0)
    out = np.zeros((len(targets), 2), dtype=np.float64)
    seg_idx = 0
    for i, target in enumerate(targets):
        while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        if seg_idx + 1 >= len(cumulative):
            out[i] = xy[-1]
            continue
        span = cumulative[seg_idx + 1] - cumulative[seg_idx]
        if span <= _EPS:
            out[i] = xy[seg_idx]
            continue
        ratio = (target - cumulative[seg_idx]) / span
        out[i] = (1.0 - ratio) * xy[seg_idx] + ratio * xy[seg_idx + 1]
    return out


def _heading_from_xy(xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(xy) == 1:
        return np.zeros((1,), dtype=np.float64)
    delta = np.zeros_like(xy)
    delta[0] = xy[1] - xy[0]
    delta[-1] = xy[-1] - xy[-2]
    if len(xy) > 2:
        delta[1:-1] = xy[2:] - xy[:-2]
    return np.arctan2(delta[:, 1], delta[:, 0])


def _densify_xy(points: Sequence[Sequence[float]], spacing_m: float) -> npt.NDArray[np.float64]:
    xy = _as_xy_array(points)
    if len(xy) <= 1:
        return xy
    cumulative = _cumulative_lengths(xy)
    total = float(cumulative[-1])
    if total <= spacing_m + _EPS:
        return xy
    targets = np.arange(0.0, total, spacing_m, dtype=np.float64)
    if targets.size == 0 or targets[-1] < total - _EPS:
        targets = np.append(targets, total)
    return _sample_polyline_at_progress(xy, cumulative, targets)


def _densify_xyh(points: Sequence[Sequence[float]], spacing_m: float) -> List[StateSE2]:
    dense_xy = _densify_xy(points, spacing_m)
    headings = _heading_from_xy(dense_xy)
    return [StateSE2(float(x), float(y), float(h)) for (x, y), h in zip(dense_xy, headings)]


def _lane_polygon(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Polygon:
    left_xy = _as_xy_array(left)
    right_xy = _as_xy_array(right)
    ring = np.vstack([left_xy, right_xy[::-1]])
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def _buffered_linestring_polygon(points: Sequence[Sequence[float]], width: float) -> Polygon:
    line = LineString(_as_xy_array(points))
    polygon = line.buffer(width, cap_style=2, join_style=2)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


class ApolloPath(PolylineMapObject):
    def __init__(self, path_id: str, points: Sequence[Sequence[float]], spacing_m: float) -> None:
        super().__init__(path_id)
        self._discrete_path = _densify_xyh(points, spacing_m)
        if not self._discrete_path:
            self._discrete_path = [StateSE2(0.0, 0.0, 0.0)]
        self._xy = np.array([[state.x, state.y] for state in self._discrete_path], dtype=np.float64)
        self._cumulative = _cumulative_lengths(self._xy)
        self._linestring = LineString(self._xy)

    @property
    def linestring(self) -> LineString:
        return self._linestring

    @property
    def length(self) -> float:
        return float(self._cumulative[-1]) if len(self._cumulative) else 0.0

    @property
    def discrete_path(self) -> List[StateSE2]:
        return self._discrete_path

    def project(self, point: Point2D | Point | Sequence[float]) -> float:
        return float(self._linestring.project(_shapely_point(point)))

    def interpolate(self, progress: float) -> StateSE2:
        clamped = float(np.clip(progress, 0.0, self.length))
        sample_xy = _sample_polyline_at_progress(
            self._xy,
            self._cumulative,
            np.array([clamped], dtype=np.float64),
        )[0]
        heading = float(self.get_nearest_pose_from_position(Point2D(sample_xy[0], sample_xy[1])).heading)
        return StateSE2(float(sample_xy[0]), float(sample_xy[1]), heading)

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        return self.project(point)

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        progress = self.project(point)
        if len(self._xy) == 1:
            return self._discrete_path[0]
        idx = int(np.argmin(np.abs(self._cumulative - progress)))
        return self._discrete_path[idx]

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        return 0.0


class ApolloPolygonObject(PolygonMapObject):
    def __init__(self, object_id: str, polygon: Polygon, original_id: str) -> None:
        super().__init__(object_id)
        self.original_id = str(original_id)
        self._polygon = polygon

    @property
    def polygon(self) -> Polygon:
        return self._polygon


class ApolloRoadblock(RoadBlockGraphEdgeMapObject):
    def __init__(
        self,
        object_id: str,
        original_id: str,
        lane_polygons: List[Polygon],
        is_connector: bool,
    ) -> None:
        super().__init__(object_id)
        self.original_id = str(original_id)
        self.is_connector = bool(is_connector)
        self._lane_polygons = lane_polygons
        self._polygon: Optional[Polygon] = None
        self._interior_edges: List[ApolloLane] = []
        self._incoming_edges: List[ApolloRoadblock] = []
        self._outgoing_edges: List[ApolloRoadblock] = []

    @property
    def polygon(self) -> Polygon:
        if self._polygon is None:
            polygon = unary_union(self._lane_polygons)
            if hasattr(polygon, "geoms"):
                polygon = unary_union([geom for geom in polygon.geoms])
            self._polygon = polygon
        return self._polygon

    def fast_distance_to_point(self, point: Point) -> float:
        return float(min(polygon.distance(point) for polygon in self._lane_polygons))

    @property
    def incoming_edges(self) -> List[ApolloRoadblock]:
        return self._incoming_edges

    @property
    def outgoing_edges(self) -> List[ApolloRoadblock]:
        return self._outgoing_edges

    @property
    def interior_edges(self) -> List["ApolloLane"]:
        return self._interior_edges

    @property
    def parallel_edges(self) -> List["ApolloRoadblock"]:
        return [self]

    @property
    def children_stop_lines(self) -> List:
        return []


class ApolloLane(LaneGraphEdgeMapObject):
    def __init__(
        self,
        object_id: str,
        original_id: str,
        center_points: Sequence[Sequence[float]],
        left_points: Sequence[Sequence[float]],
        right_points: Sequence[Sequence[float]],
        polygon: Polygon,
        speed_limit_mps: Optional[float],
        original_left_neighbors: List[str],
        original_right_neighbors: List[str],
        signal_overlap_lane_ids: set[str],
        search_xy: npt.NDArray[np.float64],
    ) -> None:
        super().__init__(object_id)
        self.original_id = str(original_id)
        self._center_points = center_points
        self._left_points = left_points
        self._right_points = right_points
        self._baseline_path: Optional[ApolloPath] = None
        self._left_boundary: Optional[ApolloPath] = None
        self._right_boundary: Optional[ApolloPath] = None
        self._polygon = polygon
        self._speed_limit_mps = None if speed_limit_mps is None else float(speed_limit_mps)
        self._original_left_neighbors = [str(v) for v in original_left_neighbors]
        self._original_right_neighbors = [str(v) for v in original_right_neighbors]
        self._signal_overlap_lane_ids = signal_overlap_lane_ids
        self._search_xy = search_xy
        self._incoming_edges: List[ApolloLane] = []
        self._outgoing_edges: List[ApolloLane] = []
        self._parent: Optional[ApolloRoadblock] = None
        self._left_adjacent: Optional[ApolloLane] = None
        self._right_adjacent: Optional[ApolloLane] = None

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def incoming_edges(self) -> List["ApolloLane"]:
        return self._incoming_edges

    @property
    def outgoing_edges(self) -> List["ApolloLane"]:
        return self._outgoing_edges

    @property
    def parallel_edges(self) -> List["ApolloLane"]:
        if self._parent is None:
            return [self]
        return list(self._parent.interior_edges)

    @property
    def baseline_path(self) -> ApolloPath:
        if self._baseline_path is None:
            self._baseline_path = ApolloPath(f"{self.id}:baseline", self._center_points, _BASELINE_SPACING_M)
        return self._baseline_path

    @property
    def left_boundary(self) -> ApolloPath:
        if self._left_boundary is None:
            self._left_boundary = ApolloPath(f"{self.id}:left", self._left_points, _BASELINE_SPACING_M)
        return self._left_boundary

    @property
    def right_boundary(self) -> ApolloPath:
        if self._right_boundary is None:
            self._right_boundary = ApolloPath(f"{self.id}:right", self._right_points, _BASELINE_SPACING_M)
        return self._right_boundary

    @property
    def speed_limit_mps(self) -> Optional[float]:
        return self._speed_limit_mps

    def get_roadblock_id(self) -> str:
        if self._parent is None:
            raise RuntimeError(f"Lane {self.id} has no parent roadblock")
        return self._parent.id

    def parent(self) -> ApolloRoadblock:
        if self._parent is None:
            raise RuntimeError(f"Lane {self.id} has no parent roadblock")
        return self._parent

    def has_traffic_lights(self) -> bool:
        return self.original_id in self._signal_overlap_lane_ids

    @property
    def stop_lines(self) -> List:
        return []

    def is_left_of(self, other: "ApolloLane") -> bool:
        assert self.get_roadblock_id() == other.get_roadblock_id()
        current = self
        seen = {self.id}
        while current._right_adjacent is not None and current._right_adjacent.id not in seen:
            if current._right_adjacent.id == other.id:
                return True
            seen.add(current._right_adjacent.id)
            current = current._right_adjacent
        return False

    def is_right_of(self, other: "ApolloLane") -> bool:
        return other.is_left_of(self)

    @property
    def adjacent_edges(self) -> Tuple[Optional["ApolloLane"], Optional["ApolloLane"]]:
        return self._left_adjacent, self._right_adjacent

    def get_width_left_right(self, point: Point2D, include_outside: bool = False) -> Tuple[float, float]:
        shp = _shapely_point(point)
        return float(self._left_boundary.linestring.distance(shp)), float(
            self._right_boundary.linestring.distance(shp)
        )

    def oriented_distance(self, point: Point2D) -> float:
        x, y = _point_xy(point)
        pose = self._baseline_path.get_nearest_pose_from_position(point)
        dx = x - pose.x
        dy = y - pose.y
        left_normal = np.array([-math.sin(pose.heading), math.cos(pose.heading)], dtype=np.float64)
        signed = dx * left_normal[0] + dy * left_normal[1]
        return float(signed)


class ApolloMap(AbstractMap):
    def __init__(self, map_path: str | Path, map_name: Optional[str] = None) -> None:
        self._map_path = Path(map_path)
        payload = json.loads(self._map_path.read_text())
        self._map_name = str(map_name or self._map_path.parent.name)
        self._payload = payload

        self._public_by_original: Dict[Tuple[str, str], str] = {}
        self._original_by_public: Dict[str, Tuple[str, str]] = {}
        self._alias_by_layer: Dict[SemanticMapLayer, Dict[str, object]] = {}

        self._signal_overlap_lane_ids = {
            str(lane_id)
            for signal in payload.get("signals", {}).values()
            for lane_id in signal.get("overlap_lane_ids", [])
        }

        self._lanes_by_original: Dict[str, ApolloLane] = {}
        self._lanes_by_public: Dict[str, ApolloLane] = {}
        self._roadblocks_by_original: Dict[str, ApolloRoadblock] = {}
        self._roadblocks_by_public: Dict[str, ApolloRoadblock] = {}
        self._crosswalks: Dict[str, ApolloPolygonObject] = {}
        self._signals: Dict[str, ApolloPolygonObject] = {}

        self._build_id_mapping(payload)
        self._build_objects(payload)
        self._build_indices()

    @property
    def map_name(self) -> str:
        return self._map_name

    def to_public_id(self, namespace: str, original_id: str) -> str:
        key = (namespace, str(original_id))
        if key not in self._public_by_original:
            raise KeyError(f"Unknown {namespace} id: {original_id}")
        return self._public_by_original[key]

    def to_original_id(self, public_id: str) -> str:
        namespace, original = self._original_by_public[str(public_id)]
        return original

    def to_public_lane_id(self, original_id: str) -> str:
        return self.to_public_id("lane", original_id)

    def to_public_roadblock_id(self, original_id: str) -> str:
        return self.to_public_id("roadblock", original_id)

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        return [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.ROADBLOCK_CONNECTOR,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.TRAFFIC_LIGHT,
        ]

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        return []

    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        raise NotImplementedError("ApolloMap does not provide raster layers")

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        raise NotImplementedError("ApolloMap does not provide raster maps")

    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[object]:
        shp = _shapely_point(point)
        return [obj for obj in self._layer_objects(layer) if self._object_contains(obj, shp)]

    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[object]:
        objects = self.get_all_map_objects(point, layer)
        assert len(objects) <= 1
        return objects[0] if objects else None

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        return self.get_one_map_object(point, layer) is not None

    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[SemanticMapLayer]
    ) -> Dict[SemanticMapLayer, List[object]]:
        center = _shapely_point(point)
        query_geom = center.buffer(float(radius))
        result: Dict[SemanticMapLayer, List[object]] = {}
        for layer in layers:
            if layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
                candidates = self._query_lane_index(layer, query_geom, center, float(radius))
            else:
                candidates = self._query_polygon_index(layer, query_geom, center, float(radius))
            result[layer] = candidates
        return result

    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> Optional[object]:
        return self._alias_by_layer.get(layer, {}).get(str(object_id))

    def get_distance_to_nearest_map_object(
        self, point: Point2D, layer: SemanticMapLayer
    ) -> Tuple[Optional[str], Optional[float]]:
        center = _shapely_point(point)
        objects = self._layer_objects(layer)
        if not objects:
            return None, math.nan
        best_obj = min(objects, key=lambda obj: self._object_distance(obj, center))
        return best_obj.id, float(self._object_distance(best_obj, center))

    def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
        raise NotImplementedError("ApolloMap does not provide raster layers")

    def get_distances_matrix_to_nearest_map_object(
        self, points: List[Point2D], layer: SemanticMapLayer
    ) -> Optional[npt.NDArray[np.float64]]:
        distances = [self.get_distance_to_nearest_map_object(point, layer)[1] for point in points]
        return np.asarray(distances, dtype=np.float64)

    def initialize_all_layers(self) -> None:
        return None

    def get_nearest_lane(self, point: Point2D, include_connectors: bool = True) -> Tuple[Optional[ApolloLane], float]:
        candidates = list(self._layer_objects(SemanticMapLayer.LANE))
        if include_connectors:
            candidates.extend(self._layer_objects(SemanticMapLayer.LANE_CONNECTOR))
        if not candidates:
            return None, math.nan
        center = _shapely_point(point)
        lane = min(candidates, key=lambda obj: obj.baseline_path.linestring.distance(center))
        return lane, float(lane.baseline_path.linestring.distance(center))

    def _get_roadblock(self, object_id: str) -> Optional[ApolloRoadblock]:
        obj = self.get_map_object(object_id, SemanticMapLayer.ROADBLOCK)
        if obj is None:
            return None
        return obj  # type: ignore[return-value]

    def _get_roadblock_connector(self, object_id: str) -> Optional[ApolloRoadblock]:
        obj = self.get_map_object(object_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if obj is None:
            return None
        return obj  # type: ignore[return-value]

    def _build_id_mapping(self, payload: dict) -> None:
        items: List[Tuple[str, str]] = []
        items.extend(("lane", str(lane["id"])) for lane in payload.get("lanes", []))
        items.extend(("roadblock", str(rb_id)) for rb_id in payload.get("roadblocks", {}).keys())
        items.extend(("crosswalk", str(cw_id)) for cw_id in payload.get("crosswalks", {}).keys())
        items.extend(("signal", str(sig_id)) for sig_id in payload.get("signals", {}).keys())
        for idx, key in enumerate(sorted(items), start=1):
            public_id = str(idx)
            self._public_by_original[key] = public_id
            self._original_by_public[public_id] = key

    def _build_objects(self, payload: dict) -> None:
        roadblocks = payload.get("roadblocks", {})
        lane_to_roadblock = {str(k): str(v) for k, v in payload.get("lane_to_roadblock", {}).items()}

        for lane_payload in payload.get("lanes", []):
            original_id = str(lane_payload["id"])
            public_id = self.to_public_lane_id(original_id)
            polygon = _lane_polygon(lane_payload["left"], lane_payload["right"])
            lane = ApolloLane(
                object_id=public_id,
                original_id=original_id,
                center_points=lane_payload["central"],
                left_points=lane_payload["left"],
                right_points=lane_payload["right"],
                polygon=polygon,
                speed_limit_mps=lane_payload.get("speed_limit"),
                original_left_neighbors=lane_payload.get("left_neighbor_ids", []),
                original_right_neighbors=lane_payload.get("right_neighbor_ids", []),
                signal_overlap_lane_ids=self._signal_overlap_lane_ids,
                search_xy=_densify_xy(lane_payload["central"], _LANE_INDEX_SPACING_M),
            )
            self._lanes_by_original[original_id] = lane
            self._lanes_by_public[public_id] = lane

        for original_id, rb_payload in roadblocks.items():
            public_id = self.to_public_roadblock_id(original_id)
            lane_polygons = [self._lanes_by_original[str(lane_id)].polygon for lane_id in rb_payload.get("lane_ids", [])]
            roadblock = ApolloRoadblock(
                object_id=public_id,
                original_id=original_id,
                lane_polygons=lane_polygons,
                is_connector=bool(rb_payload.get("junction_id")),
            )
            self._roadblocks_by_original[original_id] = roadblock
            self._roadblocks_by_public[public_id] = roadblock

        for lane_original_id, lane in self._lanes_by_original.items():
            parent_original_id = lane_to_roadblock[lane_original_id]
            lane._parent = self._roadblocks_by_original[parent_original_id]
            lane._parent._interior_edges.append(lane)
            lane._incoming_edges = [
                self._lanes_by_original[str(pred_id)]
                for pred_id in payload["lanes"][self._lane_payload_index(payload, lane_original_id)].get("predecessor_ids", [])
                if str(pred_id) in self._lanes_by_original
            ]
            lane._outgoing_edges = [
                self._lanes_by_original[str(succ_id)]
                for succ_id in payload["lanes"][self._lane_payload_index(payload, lane_original_id)].get("successor_ids", [])
                if str(succ_id) in self._lanes_by_original
            ]

        for lane_original_id, lane in self._lanes_by_original.items():
            lane._left_adjacent = self._resolve_first_neighbor(lane._original_left_neighbors)
            lane._right_adjacent = self._resolve_first_neighbor(lane._original_right_neighbors)

        for roadblock in self._roadblocks_by_original.values():
            incoming: Dict[str, ApolloRoadblock] = {}
            outgoing: Dict[str, ApolloRoadblock] = {}
            for lane in roadblock.interior_edges:
                for prev_lane in lane.incoming_edges:
                    parent = prev_lane.parent()
                    if parent.id != roadblock.id:
                        incoming[parent.id] = parent
                for next_lane in lane.outgoing_edges:
                    parent = next_lane.parent()
                    if parent.id != roadblock.id:
                        outgoing[parent.id] = parent
            roadblock._incoming_edges = list(incoming.values())
            roadblock._outgoing_edges = list(outgoing.values())

        for original_id, polygon_points in payload.get("crosswalks", {}).items():
            public_id = self.to_public_id("crosswalk", str(original_id))
            polygon = Polygon(_as_xy_array(polygon_points))
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            self._crosswalks[public_id] = ApolloPolygonObject(public_id, polygon, str(original_id))

        for original_id, signal_payload in payload.get("signals", {}).items():
            public_id = self.to_public_id("signal", str(original_id))
            polygon = _buffered_linestring_polygon(signal_payload["stop_line"], 0.5)
            self._signals[public_id] = ApolloPolygonObject(public_id, polygon, str(original_id))

        self._register_aliases()

    def _build_indices(self) -> None:
        return None

    def _register_aliases(self) -> None:
        self._alias_by_layer = {
            SemanticMapLayer.LANE: {},
            SemanticMapLayer.LANE_CONNECTOR: {},
            SemanticMapLayer.ROADBLOCK: {},
            SemanticMapLayer.ROADBLOCK_CONNECTOR: {},
            SemanticMapLayer.CROSSWALK: {},
            SemanticMapLayer.TRAFFIC_LIGHT: {},
        }
        for lane in self._lanes_by_public.values():
            layer = self._lane_layer(lane)
            self._alias_by_layer[layer][lane.id] = lane
            self._alias_by_layer[layer][lane.original_id] = lane
        for roadblock in self._roadblocks_by_public.values():
            layer = self._roadblock_layer(roadblock)
            self._alias_by_layer[layer][roadblock.id] = roadblock
            self._alias_by_layer[layer][roadblock.original_id] = roadblock
        for obj in self._crosswalks.values():
            self._alias_by_layer[SemanticMapLayer.CROSSWALK][obj.id] = obj
            self._alias_by_layer[SemanticMapLayer.CROSSWALK][obj.original_id] = obj
        for obj in self._signals.values():
            self._alias_by_layer[SemanticMapLayer.TRAFFIC_LIGHT][obj.id] = obj
            self._alias_by_layer[SemanticMapLayer.TRAFFIC_LIGHT][obj.original_id] = obj

    def _layer_objects(self, layer: SemanticMapLayer) -> List[object]:
        if layer == SemanticMapLayer.LANE:
            return [lane for lane in self._lanes_by_public.values() if self._lane_layer(lane) == layer]
        if layer == SemanticMapLayer.LANE_CONNECTOR:
            return [lane for lane in self._lanes_by_public.values() if self._lane_layer(lane) == layer]
        if layer == SemanticMapLayer.ROADBLOCK:
            return [rb for rb in self._roadblocks_by_public.values() if self._roadblock_layer(rb) == layer]
        if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
            return [rb for rb in self._roadblocks_by_public.values() if self._roadblock_layer(rb) == layer]
        if layer == SemanticMapLayer.CROSSWALK:
            return list(self._crosswalks.values())
        if layer == SemanticMapLayer.TRAFFIC_LIGHT:
            return list(self._signals.values())
        return []

    def _lane_layer(self, lane: ApolloLane) -> SemanticMapLayer:
        return SemanticMapLayer.LANE_CONNECTOR if lane.parent().is_connector else SemanticMapLayer.LANE

    def _roadblock_layer(self, roadblock: ApolloRoadblock) -> SemanticMapLayer:
        return SemanticMapLayer.ROADBLOCK_CONNECTOR if roadblock.is_connector else SemanticMapLayer.ROADBLOCK

    def _resolve_first_neighbor(self, neighbor_ids: Iterable[str]) -> Optional[ApolloLane]:
        for neighbor_id in neighbor_ids:
            neighbor = self._lanes_by_original.get(str(neighbor_id))
            if neighbor is not None:
                return neighbor
        return None

    def _lane_payload_index(self, payload: dict, lane_original_id: str) -> int:
        if not hasattr(self, "_lane_payload_indices"):
            self._lane_payload_indices = {
                str(lane["id"]): idx for idx, lane in enumerate(payload.get("lanes", []))
            }
        return self._lane_payload_indices[lane_original_id]

    def _query_lane_index(
        self,
        layer: SemanticMapLayer,
        query_geom: Polygon,
        center: Point,
        radius: float,
    ) -> List[ApolloLane]:
        found: List[ApolloLane] = []
        cx, cy = float(center.x), float(center.y)
        radius_sq = (radius + _EPS) ** 2
        for lane in self._layer_objects(layer):
            if lane.polygon.distance(center) <= radius + _EPS:
                found.append(lane)
                continue
            if lane._search_xy.size == 0:
                continue
            delta = lane._search_xy - np.array([cx, cy], dtype=np.float64)
            if float(np.min(np.sum(delta * delta, axis=1))) <= radius_sq:
                found.append(lane)
        return found

    def _query_polygon_index(
        self,
        layer: SemanticMapLayer,
        query_geom: Polygon,
        center: Point,
        radius: float,
    ) -> List[object]:
        found: List[object] = []
        for obj in self._layer_objects(layer):
            if self._object_distance(obj, center) <= radius + _EPS:
                found.append(obj)
        return found

    @staticmethod
    def _object_distance(obj: object, center: Point) -> float:
        if isinstance(obj, ApolloRoadblock):
            return obj.fast_distance_to_point(center)
        return float(obj.polygon.distance(center))

    @staticmethod
    def _object_contains(obj: object, center: Point) -> bool:
        if isinstance(obj, ApolloRoadblock):
            return any(polygon.contains(center) for polygon in obj._lane_polygons)
        return bool(obj.polygon.contains(center))
