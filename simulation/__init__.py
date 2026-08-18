"""simulation — planning을 실주행 로그로 재현·검증하는 도메인.

구성 (ABC + registry 추상화):
  renderers/  Renderer ABC + registry (matplotlib | nuplan)
  drivers/    EgoDriver ABC + registry (log_replay=open_loop | model_driven=closed_loop)
  render_sim.py  조립 전용 드라이버 (root config 하나로 실행)
"""
