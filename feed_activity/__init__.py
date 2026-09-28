"""
넙치 먹이활성도 실시간 산출 + RTMP 스트리밍 로직을 기능별로 나눈 패키지.

모듈 구성:
    camera        - USB(V4L2) 카메라 오픈 및 FPS 실측
    streaming     - GStreamer 기반 RTMP 송출 writer
    activity      - 프레임 특징 계산(optical flow/전경비율), 실시간 정규화,
                    YOLO 트래킹 기반 개체 이동량 누적, 먹이활성도 점수 계산,
                    표본 간 급락 방지(ScoreDropLimiter)
    overlay       - 송출 프레임에 bbox / 점수 오버레이 그리기
    logging_utils - CSV 로그 기록 + 구간별 처리시간 프로파일링
    pipeline      - 안전 종료 처리 + 위 모듈들을 엮은 메인 루프 (run)

커맨드라인 인자 정의(parse_args)는 이 패키지 밖, 엔트리 스크립트인
feed_activity_jetson_rtsp.py에 있음 (서브커맨드 없는 단일 CLI라
따로 뺄 실익이 없어 엔트리 파일에 그대로 둠)
"""
