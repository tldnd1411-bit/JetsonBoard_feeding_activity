"""안전 종료 처리 + 메인 루프: 카메라 -> YOLO 추론/추적 -> 먹이활성도 산출 -> 오버레이 -> RTMP 송출."""
import signal
import time

import cv2
from ultralytics import YOLO

from .activity import (
    MovementAccumulator,
    RunningNormalizer,
    ScoreDropLimiter,
    compute_feeding_score,
    compute_pair_features,
    ema_update,
    preprocess_frame,
)
from .camera import measure_actual_fps, open_camera
from .logging_utils import FeedingCsvLogger, StageProfiler
from .overlay import draw_detection_boxes, draw_feeding_overlay
from .streaming import open_rtmp_writer

PROFILE_EVERY = 60

# --- 모니터 없는 Jetson 환경을 고려한 안전 종료 ---------------------------
# install_signal_handlers()를 엔트리 포인트에서 한 번 호출해두면 Ctrl+C(SIGINT)
# / SIGTERM 수신 시 _STOP 플래그만 켜지고, 아래 메인 루프가 while not _STOP:
# 로 매 프레임 이를 확인해 현재 프레임까지 안전하게 마무리한 뒤 종료
_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    _STOP = True


def install_signal_handlers():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def run(args):
    # 1) 카메라 오픈
    cap = open_camera(args.camera, args.width, args.height, args.fps, args.fourcc)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height

    if args.flip_180:
        print("[CAMERA] --flip-180 지정됨: 프레임을 180도 회전해서 사용합니다.")

    reported_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    actual_fps = measure_actual_fps(cap) or reported_fps
    print(f"[CAMERA] 보고된 FPS: {reported_fps:.1f}  실측 FPS: {actual_fps:.1f}")

    # 2) 무거운 YOLO 모델 로딩을 RTMP 연결보다 먼저 끝냄
    print(f"[MODEL] 로드 중: {args.model}")
    yolo_model = YOLO(args.model)

    # 2-1) CUDA 커널 컴파일 등으로 첫 추론이 느릴 수 있어 워밍업.
    #      (track() 대신 일반 추론으로 워밍업해서, 워밍업 프레임이 track id를
    #       오염시켜 첫 구간의 이동량이 잘못 누적되는 걸 방지)
    ok, warmup_frame = cap.read()
    if ok:
        if args.flip_180:
            warmup_frame = cv2.rotate(warmup_frame, cv2.ROTATE_180)
        infer_warmup = preprocess_frame(warmup_frame, args.imgsz)
        yolo_model(infer_warmup, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
        print("[MODEL] 워밍업 완료")

    movement_acc = MovementAccumulator()
    frame_counter = 0

    # bbox 오버레이용: 추론 해상도(infer_size)는 카메라 해상도가 고정이므로 한 번만 계산
    infer_w = args.imgsz
    infer_h = int(h * args.imgsz / w) if w > 0 else args.imgsz
    infer_size = (infer_w, infer_h)
    last_detections = []  # 다음 추적 프레임이 올 때까지 마지막 검출 결과를 유지해서 그린다

    # 3) 모델 준비가 다 끝난 뒤에야 RTMP 연결을 엶
    writer = open_rtmp_writer(
        width=w, height=h, fps=actual_fps,
        rtmp_location=args.rtmp,
        bitrate=args.bitrate,
        iframeinterval=args.iframeinterval,
    )

    # 4) CSV 로그 준비
    csv_logger = FeedingCsvLogger(args.csv_dir)

    normalizer = RunningNormalizer()
    score_limiter = ScoreDropLimiter(max_drop_per_sample=args.max_score_drop_per_sample)
    prev_gray = None
    prev_sample_time = None
    ema_flow = None
    ema_fg = None
    ema_move = None
    current_score_pct = 0.0
    current_object_count = 0
    low_activity_since = None  # 먹이활성도가 임계값 이하로 내려간 시각. 계속 이하로 유지되면
    #                            그대로 두고, 임계값을 넘어서면 None으로 리셋

    t0 = time.monotonic()
    start_time = t0
    fps_vis = 0.0
    disp_frame_idx = 0

    profiler = StageProfiler(enabled=args.profile, print_every=PROFILE_EVERY)

    print("실시간 먹이활성도 산출 + RTMP 송출 시작 (Ctrl+C로 종료 가능)")

    try:
        while not _STOP:
            with profiler.measure("read"):
                ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            if args.flip_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            now = time.monotonic()
            elapsed = now - start_time

            # --- YOLO 추적: N프레임마다 수행, SAMPLE_SEC 구간 동안 이동거리 누적 ---
            frame_counter += 1
            with profiler.measure("track"):
                if frame_counter % args.track_every_n_frames == 0:
                    infer_frame = preprocess_frame(frame, args.imgsz)
                    last_detections = movement_acc.update(
                        yolo_model, infer_frame, args.conf, args.imgsz, args.device
                    )

            # --- SAMPLE_SEC 간격으로 먹이활성도 재계산 ---
            with profiler.measure("feature"):
                if prev_sample_time is None or (now - prev_sample_time) >= args.sample_sec:
                    window_sec = args.sample_sec if prev_sample_time is not None else elapsed

                    resized = preprocess_frame(frame, args.resize_width)
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                    total_mv_px, avg_mv_px_per_object_sec, object_count = \
                        movement_acc.consume_window(window_sec)
                    current_object_count = object_count

                    if prev_gray is not None:
                        feats = compute_pair_features(prev_gray, gray)

                        ema_flow = feats["flow_intensity"] if ema_flow is None \
                            else ema_update(ema_flow, feats["flow_intensity"], args.ema_span)
                        ema_fg = feats["fg_pixel_ratio"] if ema_fg is None \
                            else ema_update(ema_fg, feats["fg_pixel_ratio"], args.ema_span)
                        ema_move = avg_mv_px_per_object_sec if ema_move is None \
                            else ema_update(ema_move, avg_mv_px_per_object_sec, args.ema_span)

                        norm_flow = normalizer.update_and_normalize("flow_intensity", ema_flow)
                        norm_fg = normalizer.update_and_normalize("fg_pixel_ratio", ema_fg)
                        norm_move = normalizer.update_and_normalize(
                            "avg_mv_px_per_object_sec", ema_move
                        )

                        raw_score_pct = compute_feeding_score(norm_flow, norm_fg, norm_move)
                        # 급락 방지: 직전 발표값 대비 한 표본(sample_sec)당
                        # --max-score-drop-per-sample(%p) 넘게는 못 내려가게 설정
                        current_score_pct = score_limiter.update(raw_score_pct)

                        csv_logger.write_row(
                            elapsed, feats, total_mv_px, avg_mv_px_per_object_sec,
                            object_count, current_score_pct, raw_score_pct,
                        )

                        # --- 저활성도 지속 시간 추적: threshold_pct 이하가
                        #     duration_sec 이상 이어지면 모델을 정상 종료 ---
                        if current_score_pct <= args.low_activity_threshold_pct:
                            if low_activity_since is None:
                                low_activity_since = now
                            elif (now - low_activity_since) >= args.low_activity_duration_sec:
                                print(
                                    f"[FEEDING] 먹이활성도 {current_score_pct:.1f}%가 "
                                    f"{args.low_activity_duration_sec:.0f}초 이상 "
                                    f"{args.low_activity_threshold_pct:.0f}% 이하로 유지됨 -> 모델 종료"
                                )
                                break
                        else:
                            low_activity_since = None

                    prev_gray = gray
                    prev_sample_time = now

            # --- 화면 표시용 FPS 계산 ---
            disp_frame_idx += 1
            if disp_frame_idx % 10 == 0:
                fps_vis = 10.0 / max(1e-3, (now - t0))
                t0 = now

            with profiler.measure("draw"):
                if args.show_bbox:
                    draw_detection_boxes(frame, last_detections, infer_size, (w, h))
                draw_feeding_overlay(frame, current_score_pct, current_object_count, fps_vis)

            with profiler.measure("encode_write"):
                writer.write(frame)

            if args.preview:
                cv2.imshow("feeding activity", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            profiler.tick()

    except KeyboardInterrupt:
        print("사용자 중단 (Ctrl+C) → 종료")

    finally:
        cap.release()
        writer.release()
        csv_logger.close()
        cv2.destroyAllWindows()
        print("[STOP] 종료")
