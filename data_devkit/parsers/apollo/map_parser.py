from collections import Counter
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from data_devkit.parsers.base import MapParser
from data_devkit.parsers.registry import register_map_parser
from data_devkit.parsers.schema import MapGraph, MapLane, MapRoadBlock, MapSignal, write_json

PROTO_ROOT = Path(__file__).resolve().parent / "proto"
if str(PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTO_ROOT))

from modules.map.proto import map_lane_pb2  # noqa: E402
from modules.map.proto import map_pb2  # noqa: E402
from modules.map.proto import map_signal_pb2  # noqa: E402


def _decode_id(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        for encoding in ("utf-8", "cp949"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _id_to_str(message) -> str:
    if message is None:
        return ""
    return _decode_id(message.id)


def _enum_name(enum_type, value: int, default: str = "") -> str:
    if value is None:
        return default
    try:
        return enum_type.Name(value)
    except ValueError:
        return default or str(value)


def _curve_points(curve) -> List[List[float]]:
    points: List[List[float]] = []
    if curve is None:
        return points
    for segment in curve.segment:
        if not segment.HasField("line_segment"):
            continue
        for point in segment.line_segment.point:
            points.append([float(point.x), float(point.y), float(point.z)])
    return points


def _polygon_points(polygon) -> List[List[float]]:
    return [[float(point.x), float(point.y)] for point in polygon.point]


def _merge_ids(*groups: Sequence[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for group in groups:
        for item in group:
            if not item or item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def _flatten_stop_line(curves) -> List[List[float]]:
    points: List[List[float]] = []
    for curve in curves:
        for xyz in _curve_points(curve):
            points.append(xyz[:2])
    return points


class ApolloMapParser(MapParser):
    def parse(self, map_path: str, out_dir: str, map_name: str) -> Dict[str, object]:
        source_path = Path(map_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Map file not found: {source_path}")

        hdmap = map_pb2.Map()
        hdmap.ParseFromString(source_path.read_bytes())

        lane_to_roadblock: Dict[str, str] = {}
        roadblocks: Dict[str, MapRoadBlock] = {}
        lane_road_section: Dict[str, Tuple[str, str, str]] = {}
        for road in hdmap.road:
            road_id = _id_to_str(road.id)
            road_junction_id = _id_to_str(road.junction_id) if road.HasField("junction_id") else ""
            for section in road.section:
                section_id = _id_to_str(section.id)
                roadblock_id = f"{road_id}#{section_id}"
                lane_ids = [_id_to_str(lane_id) for lane_id in section.lane_id]
                roadblocks[roadblock_id] = MapRoadBlock(
                    road_id=road_id,
                    section_id=section_id,
                    lane_ids=lane_ids,
                    junction_id=road_junction_id,
                )
                for lane_id in lane_ids:
                    lane_to_roadblock[lane_id] = roadblock_id
                    lane_road_section[lane_id] = (road_id, section_id, road_junction_id)

        overlap_index: Dict[str, List[str]] = {}
        for lane in hdmap.lane:
            lane_id = _id_to_str(lane.id)
            for overlap_id in lane.overlap_id:
                overlap_index.setdefault(_id_to_str(overlap_id), []).append(lane_id)

        lanes: List[MapLane] = []
        roadtype_counter: Counter = Counter()
        subturn_counter: Counter = Counter()
        for lane in hdmap.lane:
            lane_id = _id_to_str(lane.id)
            predecessor_ids = [_id_to_str(item) for item in lane.predecessor_id]
            successor_ids = [_id_to_str(item) for item in lane.successor_id]
            left_neighbor_ids = _merge_ids(
                [_id_to_str(item) for item in lane.left_neighbor_forward_lane_id],
                [_id_to_str(item) for item in lane.left_neighbor_reverse_lane_id],
            )
            right_neighbor_ids = _merge_ids(
                [_id_to_str(item) for item in lane.right_neighbor_forward_lane_id],
                [_id_to_str(item) for item in lane.right_neighbor_reverse_lane_id],
            )
            turn = _enum_name(map_lane_pb2.Lane.LaneTurn, int(lane.turn), "NO_TURN")
            subturn = _enum_name(
                map_lane_pb2.Lane.LaneSubTurn, int(lane.subturn), "DEFAULT"
            )
            roadtype = _enum_name(
                map_lane_pb2.Lane.LaneRoadType, int(lane.Roadtype), "NORMAL"
            )
            roadtype_counter[roadtype] += 1
            subturn_counter[subturn] += 1
            mapped_road_id, mapped_section_id, mapped_junction_id = lane_road_section.get(
                lane_id, ("", "", "")
            )
            junction_id = (
                _id_to_str(lane.junction_id) if lane.HasField("junction_id") else mapped_junction_id
            )
            lanes.append(
                MapLane(
                    id=lane_id,
                    central=_curve_points(lane.central_curve),
                    left=_curve_points(lane.left_boundary.curve)
                    if lane.HasField("left_boundary")
                    else [],
                    right=_curve_points(lane.right_boundary.curve)
                    if lane.HasField("right_boundary")
                    else [],
                    length=float(lane.length),
                    speed_limit=float(lane.speed_limit),
                    turn=turn,
                    subturn=subturn,
                    roadtype=roadtype,
                    predecessor_ids=predecessor_ids,
                    successor_ids=successor_ids,
                    left_neighbor_ids=left_neighbor_ids,
                    right_neighbor_ids=right_neighbor_ids,
                    junction_id=junction_id,
                    road_id=mapped_road_id,
                    section_id=mapped_section_id,
                )
            )

        crosswalks = {
            _id_to_str(crosswalk.id): _polygon_points(crosswalk.polygon)
            for crosswalk in hdmap.crosswalk
        }

        signals: Dict[str, MapSignal] = {}
        for signal in hdmap.signal:
            signal_id = _id_to_str(signal.id)
            overlap_lane_ids = _merge_ids(
                *[
                    overlap_index.get(_id_to_str(overlap_id), [])
                    for overlap_id in signal.overlap_id
                ]
            )
            signals[signal_id] = MapSignal(
                stop_line=_flatten_stop_line(signal.stop_line),
                overlap_lane_ids=overlap_lane_ids,
                subsignal_types=[
                    _enum_name(map_signal_pb2.Subsignal.Type, int(subsignal.type), "UNKNOWN")
                    for subsignal in signal.subsignal
                ],
            )

        custom_zones = {
            "tunnel": [_polygon_points(zone.polygon) for zone in hdmap.tunnel],
            "underpass": [_polygon_points(zone.polygon) for zone in hdmap.underpass],
            "no_auto_driving": [
                _polygon_points(zone.polygon) for zone in hdmap.no_auto_driving_zone
            ],
            "construction": [
                _polygon_points(zone.polygon) for zone in hdmap.construction_zone
            ],
            "alleyway": [_polygon_points(zone.polygon) for zone in hdmap.alleyway],
        }

        map_graph = MapGraph(
            proj=hdmap.header.projection.proj
            if hdmap.HasField("header") and hdmap.header.HasField("projection")
            else "",
            lanes=lanes,
            roadblocks=roadblocks,
            lane_to_roadblock=lane_to_roadblock,
            crosswalks=crosswalks,
            signals=signals,
            custom_zones=custom_zones,
        )

        output_path = Path(out_dir) / "map_graph.json"
        write_json(str(output_path), map_graph)

        return {
            "map_name": map_name,
            "map_path": os.fspath(source_path),
            "out_path": os.fspath(output_path),
            "proj": map_graph.proj,
            "counts": {
                "lanes": len(map_graph.lanes),
                "roadblocks": len(map_graph.roadblocks),
                "crosswalks": len(map_graph.crosswalks),
                "signals": len(map_graph.signals),
                "tunnel": len(map_graph.custom_zones["tunnel"]),
                "underpass": len(map_graph.custom_zones["underpass"]),
                "no_auto_driving": len(map_graph.custom_zones["no_auto_driving"]),
                "construction": len(map_graph.custom_zones["construction"]),
                "alleyway": len(map_graph.custom_zones["alleyway"]),
            },
            "distributions": {
                "roadtype": dict(sorted(roadtype_counter.items())),
                "subturn": dict(sorted(subturn_counter.items())),
            },
        }


register_map_parser("apollo", ApolloMapParser)
