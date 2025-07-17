# 프로그램 5-4 RANSAC을 이용해 호모그래피 추정하기

import cv2 as cv
import numpy as np

# --- 영상 로드 및 특징점 검출/매칭 (프로그램 5-3과 유사) ---
# 모델 영상 로드 및 크롭
img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]  # 버스를 크롭하여 모델 영상으로 사용
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# 장면 영상 로드
img2 = cv.imread('mot_color83.jpg')  # 장면 영상
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성 및 특징점/기술자 계산
# 참고: SIFT 관련 오류 발생 시 'opencv-contrib-python' 패키지 설치 확인
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# FLANN 매칭기 생성 및 KNN 매칭 수행
flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)  # 최근접 2개 찾기

# Lowe의 비율 테스트로 좋은 매칭 선별
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)
# --- 여기까지 프로그램 5-3과 거의 동일 ---

# 호모그래피 추정을 위한 매칭된 점들의 좌표 추출
# good_match 리스트에서 각 매칭(gm)에 대해,
# 모델 영상(kp1)에서의 좌표(gm.queryIdx)와 장면 영상(kp2)에서의 좌표(gm.trainIdx)를 가져옴
points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])


# findHomography 함수를 사용하여 매칭된 점들(points1, points2)로부터
# 호모그래피 행렬(H)을 계산합니다. cv.RANSAC 알고리즘을 사용하여 이상치(outlier)에 강인하게 추정합니다.
H, _ = cv.findHomography(points1, points2, cv.RANSAC)

# 각 영상의 높이(h)와 너비(w)를 가져옵니다.
h1, w1 = img1.shape[0], img1.shape[1]  # 첫 번째 영상(모델)의 크기
h2, w2 = img2.shape[0], img2.shape[1]  # 두 번째 영상(장면)의 크기

# 첫 번째 영상(모델)의 네 꼭짓점 좌표를 정의합니다.
box1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(4, 1, 2)
# 계산된 호모그래피 행렬(H)을 사용하여 모델 영상의 꼭짓점(box1)을
# 장면 영상에서의 좌표(box2)로 변환(투영)합니다.
box2 = cv.perspectiveTransform(box1, H)

# 변환된 꼭짓점(box2)을 사용하여 장면 영상(img2) 위에 초록색 사각형을 그립니다.
# 이는 장면 영상에서 모델 영상(버스)이 검출된 위치를 나타냅니다.
img2 = cv.polylines(img2, [np.int32(box2)], True, (0, 255, 0), 8) # 초록색, 두께 8

# 매칭 결과와 검출된 영역을 함께 보여줄 출력 이미지 생성
img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
# 매칭 결과(good_match)를 그림 (장면 영상 img2에는 box2가 그려진 상태)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 최종 결과 이미지 표시
cv.imshow('Matches and Homography', img_match)

k = cv.waitKey()
cv.destroyAllWindows()
