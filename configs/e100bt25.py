# ROOT config — 파이프라인 run 단위 최고봉. "무엇을 쓸지"(이름)만 적는다.
# 이름 해석: modules.planning="pluto" -> planning/configs/pluto.py,
# calibration="e100" -> calibration/configs/e100.py,
# simulation="closed_loop_nuplan" -> simulation/configs/closed_loop_nuplan.py.
# 로더: common/config.py::load_config("configs/e100bt25.py").
# 파생물 경로 규약(클립-우선): work/<clip_id>/{parsed,input,inference,sim}.

config = dict(
    clip_id="E100BT-25_20260716151711_00006",
    record="data/bag/E100BT-25/20260716151711.record.00006",
    source="apollo_record",  # common/io parser registry key
    pose="apollo_record",    # common/io pose provider registry key
    map_name="AYG",
    map_path="work/maps/AYG/map_graph.json",
    # feature ego-shape 키(학습 분포와 일치하는 pacifica 제원 사용).
    # calibration 제원의 코드 배선(rear-axle/STEER_GAIN/충돌 shape)은 다음 태스크.
    vehicle="pacifica",
    modules=dict(planning="pluto", perception=None, localization=None),
    calibration="e100",
    simulation="closed_loop_nuplan",
)
