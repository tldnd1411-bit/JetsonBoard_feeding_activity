"""송출 프레임 위에 bbox와 먹이활성도 점수를 그림"""
import cv2


def draw_detection_boxes(frame, detections, infer_size, frame_size):
    """
    YOLO 추적 결과(infer_size 좌표계의 bbox)를 실제 송출 프레임(frame_size)
    좌표로 스케일링해서 그린다.

    detections: [(track_id, x1, y1, x2, y2, conf), ...]  (infer_size 좌표계)
    infer_size: (infer_w, infer_h)  — 추론에 사용한 리사이즈 해상도
    frame_size: (frame_w, frame_h) — 실제 프레임(원본) 해상도
    """
    if not detections:
        return

    infer_w, infer_h = infer_size
    frame_w, frame_h = frame_size
    if infer_w <= 0 or infer_h <= 0:
        return
    scale_x = frame_w / infer_w
    scale_y = frame_h / infer_h

    for track_id, x1, y1, x2, y2, conf in detections:
        px1, py1 = int(x1 * scale_x), int(y1 * scale_y)
        px2, py2 = int(x2 * scale_x), int(y2 * scale_y)

        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

        label = f"ID {track_id}"
        if conf is not None:
            label += f" {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(py1, lh + 4)
        cv2.rectangle(frame, (px1, label_y - lh - 4), (px1 + lw + 4, label_y), (0, 255, 0), -1)
        cv2.putText(frame, label, (px1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_feeding_overlay(frame, score_pct, object_count, fps_vis):
    text = f"Feeding Activity: {score_pct:5.1f}%  (obj: {object_count})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    pad = 10
    cv2.rectangle(frame, (0, 0), (tw + pad * 2, th + pad * 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:{fps_vis:4.1f}", (pad, th + pad + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
