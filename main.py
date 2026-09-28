#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
넙치 먹이활성도 실시간 산출 + RTMP 송출 (USB 카메라 / V4L2, Jetson)

실제 동작 로직은 feed_activity/ 패키지에 있으며, 이 파일은 인자 정의와
엔트리 포인트만 담당합니다. feed_activity/ 폴더가 같은 위치에 있어야 합니다.

실행 예:
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0
    python feed_activity_jetson_rtsp.py \
        --tank-id tank-03 \
        --camera /dev/video0 \
        --model best.engine \
        --rtmp rtmp://192.168.10.8:19350/x/9vuEzgKkN68p
"""
import os

# ultralytics는 import 시점에 이 값을 읽어 고정하므로, pipeline import보다 먼저 와야 함.
# 켜져 있으면 .engine에 박힌 요구 버전에 맞춰 numpy/opencv를 런타임에 pip로 갈아치움.
os.environ.setdefault("YOLO_AUTOINSTALL", "0")

import argparse

import requests

from feed_activity.pipeline import install_signal_handlers, run

print("라이브러리 로드 완료")


# 같은 Jetson에서 jetson_robot_api 서버(포트 8100)가 떠 있다는 전제.
ROBOT_API_BASE = "http://localhost:8100"
MODEL_TYPE = "feeding_activity"


def notify_model_start(tank_id: str) -> None:
    try:
        requests.post(
            f"{ROBOT_API_BASE}/api/v1/model-runs/start",
            json={"model_type": MODEL_TYPE, "tank_id": tank_id},
            timeout=3,
        )
        print(f"[flag] 시작 신호 전송 완료 (tank_id={tank_id})")
    except requests.RequestException as e:
        # flag 전송 실패는 추론을 막지 않음
        print(f"[flag] 시작 신호 전송 실패 (무시하고 계속 진행): {e}")


def notify_model_complete(tank_id: str) -> None:
    try:
        requests.post(
            f"{ROBOT_API_BASE}/api/v1/model-runs/complete",
            json={"model_type": MODEL_TYPE, "tank_id": tank_id},
            timeout=3,
        )
        print(f"[flag] 종료 신호 전송 완료 (tank_id={tank_id})")
    except requests.RequestException as e:
        print(f"[flag] 종료 신호 전송 실패: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="넙치 먹이활성도 실시간 산출 + RTMP 스트리밍")
    # 수조 정보는 로봇에서 api 통신으로 받을 예정
    p.add_argument("--tank-id", dest="tank_id", type=str, required=True,
                   help="대상 수조 ID (예: tank-03)")

    # 카메라
    p.add_argument("--camera", type=str, default="0", help="카메라 인덱스 또는 /dev/videoN")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30, help="카메라 요청 FPS (실측값은 자동 보정)")
    p.add_argument("--fourcc", type=str, default="", help="픽셀 포맷 강제 지정 (예: MJPG, UYVY)")
    p.add_argument("--flip-180", dest="flip_180", action="store_true",
                   help="프레임 180도 회전")

    # YOLO 트래킹 (개체 이동량 특징 산출용)
    p.add_argument("--model", type=str, default="best.pt", help="YOLO 가중치 경로")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640, help="YOLO 추론 해상도")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--track-every-n-frames", dest="track_every_n_frames", type=int, default=30,
                   help="추적 주기. 사이 프레임은 마지막 bbox를 유지해서 그림")
    p.add_argument("--no-bbox", dest="show_bbox", action="store_false",
                   help="bbox 오버레이 끄기")
    p.set_defaults(show_bbox=True)

    # 먹이활성도 산출
    p.add_argument("--sample-sec", dest="sample_sec", type=int, default=3,
                   help="먹이활성도 재계산 주기(초)")
    p.add_argument("--resize-width", dest="resize_width", type=int, default=640,
                   help="optical flow/전경비율 계산용 축소 해상도")
    p.add_argument("--ema-span", dest="ema_span", type=int, default=10)
    p.add_argument(
        "--max-score-drop-per-sample", dest="max_score_drop_per_sample",
        type=float, default=15.0,
        help=(
            "표본 1회당 허용 하락폭(%%p). 개체가 잠깐 가려져 점수가 급락하는 것을 완충. "
            "상승은 제한 없음. 0이면 비활성화."
        ),
    )
    p.add_argument("--csv-dir", dest="csv_dir", type=str, default="realtime_feeding_output")
    p.add_argument("--low-activity-threshold-pct", dest="low_activity_threshold_pct",
                   type=float, default=30.0,
                   help="저활성도 판정 기준(%%)")
    p.add_argument("--low-activity-duration-sec", dest="low_activity_duration_sec",
                   type=float, default=10.0,
                   help="저활성도가 이 시간(초) 이상 지속되면 자동 종료")

    # RTMP 송출
    p.add_argument("--rtmp", type=str, default="rtmp://192.168.10.8:19350/x/9vuEzgKkN68p")
    p.add_argument("--bitrate", type=int, default=3_000_000)
    p.add_argument("--iframeinterval", type=int, default=60)

    p.add_argument("--preview", action="store_true", help="로컬 창 미리보기")
    p.add_argument("--no-profile", dest="profile", action="store_false",
                   help="구간별 소요시간 출력 끄기")
    p.set_defaults(profile=True)

    return p.parse_args()


if __name__ == "__main__":
    install_signal_handlers()
    args = parse_args()

    notify_model_start(args.tank_id)
    try:
        run(args)
    finally:
        # 정상 종료/예외/시그널 어느 경로로 끝나도 종료 flag를 한 번은 보냄.
        # 단 install_signal_handlers()가 os._exit()를 쓰면 이 블록은 실행되지 않음.
        notify_model_complete(args.tank_id)