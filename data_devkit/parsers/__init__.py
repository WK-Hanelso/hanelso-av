"""원본→통합 포맷 파서 (구 common/io) — ABC + registry 진입점.

SourceParser/MapParser/EgoPoseProvider ABC는 base.py, registry는 registry.py.
구현체 패키지(apollo/, pose/)는 import 시 자기 registry 키를 등록한다.
torch import 금지 (torch 없는 경량 파싱 환경에서 돌아야 함).
"""
