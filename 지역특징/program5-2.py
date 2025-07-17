# 프로그램 5-2 SIFT 검출

import cv2 as cv

img = cv.imread('mot_color70.jpg')  # 영상 읽기
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성
# 참고: SIFT 알고리즘은 특허 문제로 OpenCV 기본 빌드에 포함되지 않을 수 있습니다.
# 이 코드가 작동하지 않으면, 'opencv-contrib-python' 패키지를 설치해야 할 수 있습니다.
# (pip install opencv-contrib-python)
sift = cv.SIFT_create()

# 그레이스케일 영상에서 키포인트(특징점) 검출 및 기술자(descriptor) 계산
kp, des = sift.detectAndCompute(gray, None)

# 검출된 키포인트를 그레이스케일 영상 위에 그림
# DRAW_RICH_KEYPOINTS 플래그는 키포인트의 크기와 방향을 함께 표시합니다.
gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 영상 출력
cv.imshow('sift', gray)

k = cv.waitKey()
cv.destroyAllWindows()