# 프로그램 5-3 FLANN 라이브러리를 이용한 SIFT 매칭

import cv2 as cv
import numpy as np
import time

# 첫 번째 영상(모델) 로드 및 관심 영역(ROI) 설정
# 'mot_color70.jpg' 영상에서 버스 부분만 잘라냄 (y: 190~350, x: 440~560)
img1 = cv.imread('mot_color70.jpg')[190:350, 440:560] # 버스를 크롭하여 모델 영상으로 사용
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) # 그레이스케일로 변환

# 두 번째 영상(장면) 로드
img2 = cv.imread('mot_color83.jpg') # 장면 영상
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) # 그레이스케일로 변환

# SIFT 객체 생성
# 참고: SIFT 관련 오류 발생 시 'opencv-contrib-python' 패키지 설치 확인
sift = cv.SIFT_create()

# 각 그레이스케일 영상에서 키포인트 검출 및 기술자 계산
kp1, des1 = sift.detectAndCompute(gray1, None) # 모델 영상 특징점 및 기술자
kp2, des2 = sift.detectAndCompute(gray2, None) # 장면 영상 특징점 및 기술자

# 검출된 특징점의 개수 출력
print('특징점 개수:', len(kp1), len(kp2)) # ①

## 프로그램 5-3의 뒷부분입니다. 앞부분에서 img1, kp1, des1, img2, kp2, des2 변수가 정의되어 있어야 합니다.
# 또한 time 모듈이 import 되어 있어야 합니다.

start = time.time() # 매칭 시작 시간 기록

# FLANN 기반 매칭기 생성
# FLANN (Fast Library for Approximate Nearest Neighbors)는 대규모 데이터셋에서
# 효율적인 최근접 이웃 검색을 제공합니다.
flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

# knnMatch 수행: des1의 각 기술자에 대해 des2에서 가장 유사한 기술자 2개를 찾음 (k=2)
knn_match = flann_matcher.knnMatch(des1, des2, 2)

# 좋은 매칭점 선별 (Lowe의 비율 테스트 적용)
T = 0.7 # 임계값 (가장 가까운 이웃과 두 번째로 가까운 이웃 간의 거리 비율)
good_match = []
for nearest1, nearest2 in knn_match:
    # 첫 번째 이웃과의 거리가 두 번째 이웃과의 거리보다 T 비율 미만이면 좋은 매칭으로 간주
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1) # 좋은 매칭점 리스트에 추가

print('매칭에 걸린 시간:', time.time() - start) # ② 매칭 소요 시간 출력

# 매칭 결과를 시각화할 이미지 준비
# img1과 img2를 가로로 이어붙일 수 있는 크기의 빈 컬러 이미지 생성
img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

# 좋은 매칭 결과(good_match)를 img_match에 그림
# flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS 는 매칭되지 않은 특징점은 그리지 않음
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 매칭 결과 이미지 표시
cv.imshow('Good Matches', img_match)

k = cv.waitKey()
cv.destroyAllWindows()