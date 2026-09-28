"""먹이활성도 산출 도메인 로직: 프레임 특징 계산 -> 실시간 정규화 -> YOLO 이동량 누적 -> 최종 점수."""
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 프레임 특징 (optical flow / 전경비율)
# ---------------------------------------------------------------------------


def preprocess_frame(frame, resize_width):
    if resize_width is None or resize_width == -1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (resize_width, int(h * resize_width / w)))


def compute_pair_features(prev_gray, curr_gray, motion_threshold=1.0):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    moving = mag[mag > motion_threshold]
    flow_intensity = float(moving.mean()) if len(moving) > 0 else 0.0

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, fg_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_pixel_ratio = float(fg_mask.mean() / 255.0)

    return {"flow_intensity": flow_intensity, "fg_pixel_ratio": fg_pixel_ratio}


def ema_update(prev_val, new_val, span):
    alpha = 2.0 / (span + 1)
    return alpha * new_val + (1 - alpha) * prev_val


class RunningNormalizer:
    """실시간 min/max 정규화."""

    def __init__(self):
        self.min_val = {}
        self.max_val = {}

    def update_and_normalize(self, col, val):
        mn = min(self.min_val.get(col, val), val)
        mx = max(self.max_val.get(col, val), val)
        self.min_val[col], self.max_val[col] = mn, mx
        if mx > mn:
            return float(np.clip((val - mn) / (mx - mn), 0.0, 1.0))
        return 0.0


# ---------------------------------------------------------------------------
# YOLO 트래킹 기반 개체 이동량
# ---------------------------------------------------------------------------


class MovementAccumulator:
    """SAMPLE_SEC 구간 동안 track별 이동거리를 누적하는 상태 저장 객체."""

    def __init__(self):
        self.prev_centers = {}
        self.reset_window()

    def reset_window(self):
        self.total_mv_px = 0.0
        self.object_ids_in_window = set()

    def update(self, model, frame_for_infer, conf, imgsz, device):
        """
        추적 + 이동거리 누적을 수행하고, 화면에 그릴 수 있도록 이번 프레임에서
        검출된 박스 목록을 [(track_id, x1, y1, x2, y2, conf), ...] 형태로
        반환한다 (좌표는 frame_for_infer 좌표계 기준)
        """
        results = model.track(
            source=frame_for_infer, persist=True, verbose=False,
            conf=conf, imgsz=imgsz, device=device,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        ids = boxes.id.int().cpu().tolist()
        centers = boxes.xywh[:, :2].cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(ids)

        current_ids = set()
        detections = []
        for track_id, (cx, cy), (x1, y1, x2, y2), box_conf in zip(ids, centers, xyxy, confs):
            current_ids.add(track_id)
            self.object_ids_in_window.add(track_id)
            prev = self.prev_centers.get(track_id)
            if prev is not None:
                dist = float(np.hypot(cx - prev[0], cy - prev[1]))
                self.total_mv_px += dist
            self.prev_centers[track_id] = (cx, cy)
            detections.append((track_id, x1, y1, x2, y2, box_conf))

        for track_id in list(self.prev_centers.keys()):
            if track_id not in current_ids:
                del self.prev_centers[track_id]

        return detections

    def consume_window(self, window_sec):
        object_count = len(self.object_ids_in_window)
        total_mv_px = self.total_mv_px
        if object_count > 0 and window_sec > 0:
            avg_mv_px_per_object_sec = total_mv_px / object_count / window_sec
        else:
            avg_mv_px_per_object_sec = 0.0
        self.reset_window()
        return total_mv_px, avg_mv_px_per_object_sec, object_count


# ---------------------------------------------------------------------------
# 최종 점수: 정규화된 특징의 가중합 (0~100%)
# ---------------------------------------------------------------------------

FEEDING_SCORE_WEIGHTS = {
    "flow_intensity": 0.34,
    "fg_pixel_ratio": 0.33,
    "avg_mv_px_per_object_sec": 0.33,
}


def compute_feeding_score(norm_flow: float, norm_fg: float, norm_move: float) -> float:
    score = (
        FEEDING_SCORE_WEIGHTS["flow_intensity"] * norm_flow
        + FEEDING_SCORE_WEIGHTS["fg_pixel_ratio"] * norm_fg
        + FEEDING_SCORE_WEIGHTS["avg_mv_px_per_object_sec"] * norm_move
    )
    return score * 100.0


# ---------------------------------------------------------------------------
# 급락 방지: 표본(sample_sec) 간 점수 변화폭 제한
# ---------------------------------------------------------------------------
#
# compute_feeding_score()는 매 sample_sec(기본 3초)마다 "그 구간만의" 값으로
# 새로 계산되기 때문에, 개체가 잠깐 가려지거나 YOLO 트래킹이 한 번 튀는 것만으로도
# 바로 앞 구간이 50%였다가 다음 구간이 3%로 떨어지는 식의 비현실적인 급락이 생길 수
# 있다. 실제 섭이 활동은 그렇게 한 순간에 사라지지 않으므로, "한 표본 구간당 몇 %p
# 까지만 내려갈 수 있는지"를 제한해서 이런 스파이크를 눌러줌
#
# 상승은 제한하지 않는다 — 활동이 실제로 시작되는 시점을 늦게 반영하면 오히려
# 해로우므로, 급격한 상승은 그대로 반영하고 급격한 하락만 완충처리 함


class ScoreDropLimiter:
    """
    급격하겨 활성도가 떨어지는 것을 방지하기 위해 limit 설정
    진짜로 활동이 멈췄다면(raw 값이 계속 낮게 나오면) 다음 구간들에서도 계속
    max_drop_per_sample씩 깎이면서 결국 raw 값까지 수렴하므로, "탐지가 실제로
    멈췄는데도 영원히 안 내려가는" 문제는 생기지 않음
    """

    def __init__(self, max_drop_per_sample: float = 15.0):
        if max_drop_per_sample < 0:
            raise ValueError("max_drop_per_sample은 0 이상이어야 합니다")
        self.max_drop_per_sample = max_drop_per_sample
        self.value = None  # 마지막으로 발표한(완충된) 점수

    def update(self, raw_score_pct: float) -> float:
        if self.value is None or self.max_drop_per_sample == 0:
            # 첫 표본(비교 기준 없음) 이거나, max_drop_per_sample=0(급락 방지 끔)이면
            # clamp 없이 raw 값을 그대로 채택한다.
            self.value = raw_score_pct
            return self.value

        drop = self.value - raw_score_pct
        if drop > self.max_drop_per_sample:
            self.value -= self.max_drop_per_sample
        else:
            # 하락폭이 허용 범위 안이거나(0 이하 = 오히려 상승) 그대로 반영
            self.value = raw_score_pct
        return self.value
