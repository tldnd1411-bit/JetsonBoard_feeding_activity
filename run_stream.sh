#!/bin/bash
# run_stream.sh
sudo fuser -k /dev/video0 2>/dev/null   # 이전에 물고 있던 프로세스가 있으면 강제 종료 (없으면 에러 무시)
sleep 1                                  # 디바이스 노드가 완전히 풀릴 시간을 잠깐 줌

# ultralytics가 .engine 파일에 박힌 요구 버전(예: numpy==1.23.5)과 현재 설치된
# 버전이 다르면 실행 중에 조용히 pip install로 numpy를 바꿔치기하는 "AutoUpdate"를
# 시도합니다. 운영 중인 보드에서 매번 이게 트리거되면 numpy/opencv/albumentations
# 등의 버전이 서로 안 맞게 꼬여서 여기서 겪은 것 같은 "AttributeError: module
# 'numpy' has no attribute 'bool'" 같은 문제가 재발할 수 있어 꺼둡니다.
# (필요한 패키지 버전은 한 번 제대로 맞춰놓고 고정해서 쓰는 게 안전합니다.)
export YOLO_AUTOINSTALL=0
python main.py \
    --tank-id tank-03 \
    --camera /dev/video0 \
    --model best.engine \
    --rtmp rtmp://14.49.189.152:19350/x/V4qnNGCpsM5N