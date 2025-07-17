# 프로그램 4-8 이진 영역의 특징을 추출하는 함수 사용하기

import skimage
import numpy as np
import cv2 as cv

orig = skimage.data.horse()
# skimage.data.horse()는 True/False 값을 가지므로, 이를 반전하여 0/255 이미지로 만듭니다.
img = 255 - np.uint8(orig) * 255
cv.imshow('Horse', img) # ① 원본 이진 이미지 표시

# findContours는 이진 이미지(0 또는 255)에서 잘 동작합니다.
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# 경계선을 컬러로 표시하기 위해 그레이스케일 이미지를 BGR로 변환
img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR) # 컬러 디스플레이용 영상
# 찾은 경계선(contours)을 분홍색(255, 0, 255)으로 그림
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.imshow('Horse with contour', img2) # ② 경계선이 그려진 이미지 표시

# 가장 바깥쪽 경계선 하나만 사용 (RETR_EXTERNAL을 사용했으므로 보통 하나만 나옴)
contour = contours[0]
# 프로그램 4-8의 뒷부분입니다. 앞부분에서 contour 변수가 정의되어 있어야 합니다.

m = cv.moments(contour) # 몇 가지 특징 계산 (모멘트)
area = cv.contourArea(contour) # 면적
# 모멘트로부터 중심 좌표 계산
cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
perimeter = cv.arcLength(contour, True) # 둘레 (True: 폐곡선)
# 둥근 정도 계산 (원형성: 1에 가까울수록 원)
roundness = (4.0 * np.pi * area) / (perimeter * perimeter)
print('면적=', area, '\n중점=(', cx, ',', cy, ')', '\n둘레=', perimeter, '\n둥근 정도=',
      roundness) # ③ 계산된 특징 출력

# 새로운 컬러 디스플레이용 영상 생성 (기존 img는 흑백)
img3 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# 경계선 근사 (Douglas-Peucker 알고리즘), 오차 임계값 8
contour_approx = cv.approxPolyDP(contour, 8, True)
# 근사된 경계선을 초록색(0, 255, 0)으로 그림
cv.drawContours(img3, [contour_approx], -1, (0, 255, 0), 2)

# 볼록 껍질(Convex Hull) 계산
hull = cv.convexHull(contour)
# hull = hull.reshape(1, hull.shape[0], hull.shape[2]) # drawContours에 넣기 위한 형태 변경 (OpenCV 버전에 따라 불필요할 수 있음)
# 볼록 껍질을 파란색(0, 0, 255)으로 그림
# 참고: 최신 OpenCV에서는 reshape 없이 hull 자체를 리스트로 감싸 [hull] 형태로 넣는 것이 일반적입니다.
# 이미지에 나온대로 코딩합니다.
hull = hull.reshape(1, hull.shape[0], hull.shape[2])
cv.drawContours(img3, hull, -1, (0, 0, 255), 2)


# 경계선 근사(초록)와 볼록 껍질(파랑)이 그려진 이미지 표시
cv.imshow('Horse with line segments and convex hull', img3) # ④

cv.waitKey()
cv.destroyAllWindows()