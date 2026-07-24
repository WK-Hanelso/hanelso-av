# pluto-apollo-sim Docker

PLUTO × Apollo **closed-loop 시뮬레이션(sim) 전용** 이미지. 파싱 → 추론 → 렌더를 **단일 venv**(numpy 1.23.4)에서 수행. 학습·pth→onnx export는 이 이미지가 아니라 별도 GPU 이미지에서 하고, **onnx 아티팩트로만 연결**한다.

## 빌드
```bash
bash docker/build.sh          # IMAGE=pluto-apollo-sim:latest
```
소스가 세 곳(`/opt/nuplan-devkit`, `~/pluto_onnx`, `~/pluto_apollo`)에 흩어져 있어 `build.sh`가 임시 컨텍스트에 모아 빌드한다. 경로가 다르면:
```bash
NUPLAN_DEVKIT=/opt/nuplan-devkit PLUTO_ONNX_ROOT=/home/culee/pluto_onnx bash docker/build.sh
```

## 왜 이런 선택인지 (검증 결과)
- **단일 venv 가능**: cyber-record는 numpy를 요구하지 않음. numpy 1.23.4 + protobuf 3.20.3에서 파싱·dump_raw·추론·렌더 전부 동작 확인 → 예전 numpy2 분리와 cross-env pickle 꼼수 불필요.
- **CPU 빌드**: 추론은 onnxruntime CPU(`use_gpu=False`), torch는 feature builder용으로만 쓰여 CPU로 충분. 그래서 **2060·5880 Ada·노트북 어디서든** 동작(CUDA 불요). 참고로 torch 1.12+cu116은 5880 Ada(sm_89)에선 GPU 미지원이라, 지금도 사실상 CPU로 돌던 것.
- **학습전용 제거**: natten(cu116 컴파일)·pytorch-lightning·torchmetrics·torchvision은 렌더 import 체인에 없어서 제외.

## GPU가 꼭 필요하면
`Dockerfile` 상단 주석 참고: base를 `nvidia/cuda:11.6-cudnn8-runtime`로, torch를 cu116, `onnxruntime`→`onnxruntime-gpu`로 교체. **단 2060(sm_75) 전용**이 되고 5880 Ada와는 호환 안 됨. sim은 CPU로 충분하므로 대개 불필요.

## 실행
```bash
# 파싱: record -> raw.pkl (routing route + TL 포함)
docker run --rm -v /path/bag:/data pluto-apollo-sim \
  python scripts/dump_raw.py /data/xxx.record --map-graph /data/map_graph.pkl --out /data/raw.pkl

# 렌더(추론): NRCLS
docker run --rm -v $PWD/pluto_apollo/out:/data -v $PWD/pluto_apollo/out:/out \
  -e PLUTO_VEHICLE=E100 -e PLUTO_AGENT_MARGIN=0.15 -e PLUTO_COLLISION_SCALE=1.0 \
  pluto-apollo-sim \
  python scripts/render_closed_loop_post.py /data/raw.pkl /data/map_graph.pkl --out /out/nrcls.mp4 --steps 600
```

## 미검증 / 재구축 TODO
- **이 문서의 도커 빌드는 실제로 빌드·실행 테스트되지 않음**(이 환경엔 docker 데몬 없음). 첫 빌드 시 아래 가능성:
  - geopandas/rasterio 계열이 시스템 GDAL 요구하면 `Dockerfile`의 `libgdal-dev gdal-bin` 주석 해제.
- **하드코딩 경로**: 스크립트가 `/home/culee/pluto_onnx` 및 ONNX 모델 경로를 하드코딩 → 무수정 동작 위해 컨테이너에 동일 경로 미러링 중. 재구축 시 `PLUTO_ONNX_ROOT`/`ONNX_MODEL` env로 파라미터화하면 미러링 제거 가능(Dockerfile에 ONNX_MODEL은 이미 세팅해 둠).
- **requirements.txt**는 현재 렌더 venv freeze에서 자동 정제한 것(241개). 재구축 시 실제 import 클로저 기준으로 더 줄일 수 있음.
