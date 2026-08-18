"""data_devkit — 데이터 생산(파서)·계약(contract)·검증의 최상위 모듈.

원칙 (C-SWM-023)
----------------
- devkit은 모델을 모른다: 모델별 분기 금지. 모델은 자기 dataloader에서
  REQUIRES(아티팩트 이름 목록)를 선언하고 :mod:`data_devkit.contract` 의
  ``check()`` 로 존재/스키마를 검증한다.
- torch import 금지: torch 없는 경량 파싱 환경에서도 import된다.
- 하위 구조:
  - :mod:`data_devkit.parsers`  — 원본→통합 포맷 파서 (구 common/io)
  - :mod:`data_devkit.contract` — 아티팩트 registry + 존재/스키마 check
"""
