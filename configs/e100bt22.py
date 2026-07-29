# ROOT config — E100BT-22 클립 (SEL 맵). 형식 설명은 configs/e100bt25.py 참조.

config = dict(
    clip_id="E100BT-22_20260716031426_00014",
    record="data/bag/E100BT-22/20260716031426.record.00014",
    source="apollo_record",
    pose="apollo_record",
    map_name="SEL",
    map_path="work/maps/SEL/map_graph.json",
    vehicle="pacifica",
    modules=dict(planning="pluto", perception=None, localization=None),
    calibration="e100",
    simulation="closed_loop_nuplan",
)
