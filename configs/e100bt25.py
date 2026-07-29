# ROOT config — 파이프라인 run 단위 최고봉. "무엇을 쓸지"(이름)만 적는다.
# 이름 해석: modules.planning="pluto" -> planning/configs/pluto.py,
# calibration="e100" -> calibration/configs/e100.py,
# simulation="closed_loop_nuplan" -> simulation/configs/closed_loop_nuplan.py.
# 로더: common/config.py::load_config("configs/e100bt25.py").
# 파생물 경로 규약(클립-우선): work/<clip_id>/{parsed,input,inference,sim}.

config = dict(
    clip_id="E100BT-25_20260716151711_00006",
    record="data/bag/E100BT-25/20260716151711.record.00006",
    source="apollo_record",  # data_devkit/parsers parser registry key
    pose="apollo_record",    # data_devkit/parsers pose provider registry key
    map_name="AYG",
    map_path="work/maps/AYG/map_graph.json",
    # 데이터 provenance 축 (data_devkit/contract.py DATA_AXES):
    # 같은 아티팩트의 복수 생산자 중 무엇을 쓸지. 현재 유효 값은
    # agents="apollo_gt", prediction=None 뿐 (bevfusion/apollo는 예약).
    data=dict(agents="apollo_gt", prediction=None),
    # feature ego-shape pin은 planning/configs/pluto.py 소유.
    modules=dict(planning="pluto", perception=None, localization=None),
    calibration="e100",
    simulation="closed_loop_nuplan",
)
