# 넙치 먹이활성도 실시간 산출

Jetson 환경에서 USB 카메라 영상을 입력받아 **넙치 먹이활성도(Feeding Activity)** 를 실시간으로 계산하고, 영상 좌상단에 활성도 값을 오버레이하며, 동시에 시간대별 먹이활성도 계산 결과를 CSV 로그 파일로 저장

---
# 아키텍처
## ![AI 기반 어류 먹이활성도 추정](docs/feeding_activity_architechure.png)

## 1. 주요 기능

- USB 카메라 실시간 입력 처리
- Optical Flow 기반 움직임 강도 계산 (Frame-difference / Motion-based Activity Analysis)
- 프레임 차분 기반 전경 픽셀 비율 계산 (Frame-difference / Motion-based Activity Analysis)
- YOLO 트래킹 기반 개체별 이동량 계산 (Object Tracking-based Movement Analysis)
- EMA 기반 특징값 평활화
- 실시간 min/max 정규화 및 3가지 특징의 가중합을 통한 먹이활성도 점수(Weighted Feeding Activity Score) 산출
- 영상 좌상단에 먹이활성도 퍼센트 및 추적 중인 개체 수 오버레이
- CSV 로그 저장
- 모니터가 없는 Jetson 환경을 고려하여 `Ctrl+C`로 종료

---

## 2. 실행 환경

### 하드웨어

- NVIDIA Jetson 보드
- USB 카메라

### 소프트웨어

- Python 3.x
- OpenCV
- NumPy
- ultralytics (YOLO `.track()` API 사용)

필요 패키지:

```bash
pip install -r requirements.txt
```

Jetson 환경에서는 OpenCV가 카메라 및 비디오 인코딩을 제대로 지원하는지 확인 필요

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

ultralytics/torch가 Jetson의 CUDA를 정상적으로 잡는지도 함께 확인 필요

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. 파일 구조 예시

```text
project/
├── feed_activity_Jetson.py
├── best.pt                          <- 넙치 탐지/추적용 YOLO 학습 가중치
├── README.md
└── realtime_feeding_output/
    ├── feeding_log_YYYYMMDD_HHMMSS.csv
```

---

## 4. 설정값

코드 상단의 설정 영역에서 카메라 입력, 해상도, FPS, 샘플링 간격, YOLO 추적 관련 값을 수정할 수 있음

```python
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS    = 30

SAMPLE_SEC   = 3
RESIZE_WIDTH = 640
EMA_SPAN = 10

YOLO_MODEL_PATH = "best.pt"
YOLO_CONF_THRESH = 0.25
YOLO_IMGSZ = RESIZE_WIDTH
YOLO_TRACK_EVERY_N_FRAMES = 2

FEEDING_SCORE_WEIGHTS = {
    "flow_intensity": -,
    "fg_pixel_ratio": -,
    "avg_mv_px_per_object_sec": -,
}
```

### 주요 설정 설명

| 설정값 | 설명 |
|---|---|
| `CAMERA_INDEX` | USB 카메라 장치 번호입니다. 일반적으로 `/dev/video0`이면 `0`을 사용 |
| `CAMERA_WIDTH` | 카메라 입력 해상도 가로 크기 |
| `CAMERA_HEIGHT` | 카메라 입력 해상도 세로 크기 |
| `CAMERA_FPS` | 카메라 요청 FPS로, 실제 FPS와 다를 수 있음 |
| `SAMPLE_SEC` | 몇 초 간격으로 먹이활성도를 재계산할지 설정 |
| `RESIZE_WIDTH` | Optical Flow/프레임 차분 특징 계산용 축소 해상도, 영상 저장은 원본 해상도로 유지됨 |
| `EMA_SPAN` | 특징값 평활화에 사용하는 EMA 스팬 |
| `YOLO_MODEL_PATH` | 넙치 탐지/추적용 YOLO 가중치 파일 경로 |
| `YOLO_CONF_THRESH` | 이 confidence 이상인 탐지만 추적에 채택 |
| `YOLO_IMGSZ` | YOLO 추론 해상도. `RESIZE_WIDTH`와 같은 스케일을 써서 이동량(px)의 단위를 Optical Flow와 맞춤 |
| `YOLO_TRACK_EVERY_N_FRAMES` | 매 N프레임마다 YOLO 추적 수행 (Jetson 연산 부담 완화 목적). 값이 너무 크면 track id 연속성이 끊겨 이동거리가 과소/과대 추정될 수 있음 |
| `FEEDING_SCORE_WEIGHTS` | 최종 점수를 만들 때 3가지 정규화된 특징(`flow_intensity`, `fg_pixel_ratio`, `avg_mv_px_per_object_sec`)에 곱하는 가중치. 합이 1이 되도록 설정 |

---

## 5. 먹이활성도 산출 방식

먹이활성도(Weighted Feeding Activity Score)는 성격이 다른 두 계열, 총 세 가지 특징값을 결합해서 계산

- Frame-difference / Motion-based Activity Analysis: `flow_intensity`, `fg_pixel_ratio`
- Object Tracking-based Movement Analysis: `avg_mv_px_per_object_sec`

### 5.1 Optical Flow 기반 움직임 강도 (`flow_intensity`)

이전 샘플 프레임과 현재 샘플 프레임 사이의 움직임을 Farneback Optical Flow로 계산

```python
flow_intensity
```

움직임 벡터의 크기 중 일정 임계값(`motion_threshold=1.0px`) 이상인 값들만 골라 그 평균을 사용. 즉 화면에서 실제로 움직이는 부분이 평균적으로 얼마나 크게/빠르게 움직였는지를 나타냄

### 5.2 프레임 차분 기반 전경 픽셀 비율 (`fg_pixel_ratio`)

이전 샘플 프레임과 현재 샘플 프레임의 절대 차이(`absdiff`)를 계산한 뒤, Otsu threshold로 자동 이진화하여 변화가 발생한 픽셀의 비율을 계산

```python
fg_pixel_ratio
```

`flow_intensity`가 "움직임의 세기"를 본다면, `fg_pixel_ratio`는 "움직임이 차지하는 면적"을 봄

### 5.3 YOLO 트래킹 기반 개체별 이동량 (`avg_mv_px_per_object_sec`)

`YOLO.track(persist=True)`로 프레임마다 개체별 track id를 얻고, 같은 id가 직전 추적 프레임 대비 얼마나 이동했는지(px)를 `SAMPLE_SEC` 구간 동안 누적. 이 방식은 오프라인 분석 스크립트가 만드는 `movement_per_second_feeding.csv`의 정의와 동일하여 같은 로직/그래프 스크립트로 바로 비교·시각화 가능

```python
total_mv_px               # 구간 동안 모든 track의 이동거리 합(px)
avg_mv_px_per_object_sec  # total_mv_px / 구간에 등장한 고유 개체 수 / 구간 길이(sec)
```

이 특징은 앞의 두 특징(움직임의 세기·면적)과 달리, 개체 단위로 "실제로 얼마나 이동했는가"를 직접 추적해서 재는 값. `YOLO_TRACK_EVERY_N_FRAMES` 프레임마다 한 번씩 추적을 수행하며, 한 프레임에서 사라졌다가 다시 나타난 track은 새 시작점부터 이동거리를 다시 누적

### 5.4 EMA 평활화

순간적인 노이즈를 줄이기 위해 세 특징값 모두에 EMA를 적용

```python
ema_flow
ema_fg
ema_move
```


## 6. 출력 파일

프로그램 실행 시 `realtime_feeding_output` 폴더가 자동 생성되며, 실행 시각 기준으로 결과 파일이 저장

### CSV 로그

파일명 예시:

```text
feeding_log_20260707_091500.csv
```

CSV 컬럼:

| 컬럼명 | 설명 |
|---|---|
| `time_sec` | 프로그램 시작 후 경과 시간 |
| `flow_intensity` | Optical Flow 기반 움직임 강도 |
| `fg_pixel_ratio` | 프레임 차분 기반 전경 픽셀 비율 |
| `total_mv_px` | 해당 구간 동안 추적된 모든 개체의 이동거리 합(px) |
| `avg_mv_px_per_object_sec` | 개체 1마리, 1초당 평균 이동거리(px) |
| `object_count` | 해당 구간에서 추적된 고유 개체 수 |
| `feeding_score_pct` | 최종 먹이활성도 점수 |


---

## 7. 실행 방법

Python 파일이 있는 경로에서 다음 명령어를 실행

```bash
./run_stream.sh
```

## 8. 종료 방법

모니터가 연결되어 있지 않은 Jetson 환경을 고려하여 키 입력 종료는 사용하지 않음


```text
Ctrl + C
```

정상 종료 시 CSV와 영상 파일이 저장됨

```text
CSV 저장 완료 : realtime_feeding_output/feeding_log_YYYYMMDD_HHMMSS.csv
```