"""GStreamer(Jetson HW 인코더)를 이용한 RTMP 송출 writer."""
import cv2


def open_rtmp_writer(width, height, fps, rtmp_location, bitrate=3_000_000, iframeinterval=60):
    """
    appsrc -> nvvidconv -> nvv4l2h264enc -> flvmux -> rtmpsink

    이 함수는 "이후부터 계속 write()가 이어지는" 시점에 호출해야 함
    (모델 로딩 등 무거운 초기화는 반드시 이 호출 전에 끝내둘 것 — 안 그러면
     연결 직후 프레임이 안 들어가서 서버가 타임아웃으로 연결을 끊을 수 있음)
    """
    pipeline = (
        "appsrc ! "
        "queue leaky=downstream max-size-buffers=4 ! "
        "videoconvert ! "
        "video/x-raw,format=(string)BGRx ! "
        "nvvidconv ! "
        "video/x-raw(memory:NVMM),format=(string)NV12 ! "
        f"nvv4l2h264enc bitrate={bitrate} insert-sps-pps=true iframeinterval={iframeinterval} ! "
        "h264parse config-interval=1 ! "
        "queue leaky=downstream max-size-buffers=8 ! "
        "flvmux streamable=true ! "
        f'rtmpsink location="{rtmp_location}" sync=false async=false'
    )

    writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"RTMP VideoWriter를 열 수 없습니다. 파이프라인:\n{pipeline}")

    print(f"[RTMP] 송출 시작: {rtmp_location}")
    print(f"[RTMP] 파이프라인: {pipeline}")
    return writer
