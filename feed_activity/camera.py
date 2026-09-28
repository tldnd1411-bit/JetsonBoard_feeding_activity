"""USB 카메라(V4L2) 오픈 및 실측 FPS 측정."""
import time

import cv2


def _decode_fourcc(value: int) -> str:
    return "".join([chr((int(value) >> 8 * i) & 0xFF) for i in range(4)])


def open_camera(camera: str, width: int, height: int, fps: int, fourcc: str = "") -> cv2.VideoCapture:
    """
    V4L2 기반으로 USB 카메라를 엽니다.

    fourcc: 카메라에 강제로 요청할 픽셀 포맷(예: 'UYVY', 'MJPG', 'YUYV').
    빈 문자열이면 드라이버의 현재 기본 포맷을 그대로 사용합니다.
    ('v4l2-ctl --device=<장치> --list-formats-ext'로 지원 포맷 확인)
    """
    source = int(camera) if camera.isdigit() else camera
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(
            f"카메라를 열 수 없습니다: {camera}\n"
            "ls -l /dev/video* 및 v4l2-ctl 결과를 확인하세요."
        )

    if fourcc:
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc.upper())
        if not cap.set(cv2.CAP_PROP_FOURCC, fourcc_code):
            print(f"[CAMERA] 경고: FourCC({fourcc}) 설정 요청이 거부되었습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 오래된 프레임 누적 최소화 (실시간성 확보)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    print(
        f"[CAMERA] device={camera}, resolution={actual_width}x{actual_height}, "
        f"fourcc={actual_fourcc}"
    )
    if (actual_width, actual_height) != (width, height):
        print(
            f"[CAMERA] 경고: 요청한 해상도({width}x{height})와 실제 협상된 해상도"
            f"({actual_width}x{actual_height})가 다릅니다. --fourcc 값을 확인하세요."
        )
    return cap


def measure_actual_fps(cap, n_frames=30, warmup_frames=5):
    """
    cap.get(cv2.CAP_PROP_FPS)는 드라이버가 '보고'하는 값이라 부정확한 경우가 많아,
    짧게 실제로 프레임을 읽어 실측 FPS를 계산
    RTMP 인코더에 넘길 fps를 정확히 맞추기 위해 사용
    """
    for _ in range(warmup_frames):
        cap.read()
    t0 = time.monotonic()
    count = 0
    for _ in range(n_frames):
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    elapsed = time.monotonic() - t0
    return count / elapsed if elapsed > 0 and count > 0 else None
