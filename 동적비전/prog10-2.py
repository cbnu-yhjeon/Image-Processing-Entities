import numpy as np
import cv2 as cv

# 비디오 파일 열기
cap = cv.VideoCapture('slow_traffic_small.mp4')

# Shi-Tomasi 코너 검출 파라미터
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# Lucas-Kanade 광류 파라미터
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 추적선을 그릴 랜덤 색상 생성
color = np.random.randint(0, 255, (100, 3))

# 첫 프레임 읽기 및 코너 검출
ret, old_frame = cap.read()  # 첫 프레임
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 물체의 이동 궤적을 그릴 영상 생성
mask = np.zeros_like(old_frame)

while (1):
    ret, frame = cap.read()
    if not ret:
        break

    new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 광류 계산 (이전 프레임, 현재 프레임, 이전 프레임의 코너)
    p1, match, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # 양호한 점 선택
    if p1 is not None:
        good_new = p1[match == 1]
        good_old = p0[match == 1]

    # 이동 궤적 그리기
    for i in range(len(good_new)):
        a, b = int(good_new[i][0]), int(good_new[i][1])
        c, d = int(good_old[i][0]), int(good_old[i][1])
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv.add(frame, mask)

    cv.imshow('LTK tracker', img)
    cv.waitKey(30)

    # 이전 프레임과 이전 코너를 현재 값으로 업데이트
    old_gray = new_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()