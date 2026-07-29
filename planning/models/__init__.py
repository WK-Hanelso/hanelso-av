"""planning/models — 모델-우선 배치. 모델 하나 = 디렉토리 하나.

모델 추가 = models/<이름>/ 디렉토리 1개 + planning/configs/<이름>.py 1개.
드라이버는 planning.interface.load_model(이름)로만 접근한다 (하드코딩 금지).
"""
