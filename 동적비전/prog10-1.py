import numpy as np
import cv2 as cv
import sys


# 광류를 시각화하는 함수
def draw_OpticalFlow(img, flow, step=16):
    for y in range(step // 2, frame.shape[0], step):
        for x in range(step // 2, frame.shape[1], step):
            dx, dy = flow[y, x].astype(np.int)
            # 움직임이 큰 곳(크기가 1 이상)은 빨간색으로, 작은 곳은 초록색으로 표시
            if dx * dx + dy * dy > 1:
                cv.line(img, (x, y), (x + dx, y + dy), (0, 0, 255), 2)
            else:
                cv.line(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2)


# 카메라와 연결 시도
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

prev = None  # 이전 프레임을 저장할 변수

while (1):
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    # 첫 프레임이면 광류 계산 없이 prev만 설정
    if prev is None:
        prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        continue

    # 현재 프레임을 그레이스케일로 변환
    curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Farneback 알고리즘으로 광류 계산
    flow = cv.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 계산된 광류를 프레임에 시각화
    draw_OpticalFlow(frame, flow)
    cv.imshow('Optical flow', frame)

    # 현재 프레임을 다음 반복을 위해 prev에 저장
    prev = curr

    key = cv.waitKey(1)  # 1밀리초 동안 키보드 입력 기다림
    if key == ord('q'):  # 'q' 키가 들어오면 루프를 빠져나감
        break

cap.release()  # 카메라와 연결을 끊음
cv.destroyAllWindows()