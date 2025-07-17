# 프로그램 5-1 해리스 특징점 검출 구현하기

import cv2 as cv
import numpy as np

# --- Part 1: 초기 설정 (이미지, 커널 정의) ---
# 테스트용 입력 영상 생성 (계단 모양의 패턴)
img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

# x, y 방향 미분 커널
ux = np.array([[-1, 0, 1]])
uy = np.array([[-1, 0, 1]]).transpose()

# 가우시안 커널 생성
k = cv.getGaussianKernel(3, 1)
# 2D 가우시안 커널 생성
g = np.outer(k, k.transpose())

# --- Part 2: 해리스 코너 계산 및 비최대 억제 ---
# 입력 영상(img)에 대해 y방향, x방향 미분 계산
dy = cv.filter2D(img, cv.CV_32F, uy)
dx = cv.filter2D(img, cv.CV_32F, ux)

# 미분값의 제곱 및 교차곱 계산
dyy = dy * dy  # I_y^2
dxx = dx * dx  # I_x^2
dyx = dy * dx  # I_y * I_x

# 구조 텐서의 각 요소에 가우시안 가중치 적용
gdyy = cv.filter2D(dyy, cv.CV_32F, g)
gdxx = cv.filter2D(dxx, cv.CV_32F, g)
gdyx = cv.filter2D(dyx, cv.CV_32F, g)

# 해리스 코너 응답 함수 C 계산 (k=0.04)
C = (gdyy * gdxx - gdyx * gdyx) - 0.04 * (gdyy + gdxx) * (gdyy + gdxx)

# 비최대 억제 (Non-maximum suppression) 및 특징점 표시
for j in range(1, C.shape[0] - 1):
    for i in range(1, C.shape[1] - 1):
        # 현재 픽셀 응답 값이 임계값(0.1)보다 크고 주변 8개 픽셀보다 크면
        if C[j, i] > 0.1 and np.sum(C[j, i] > C[j - 1:j + 2, i - 1:i + 2]) == 8:
            # 원본 이미지에 값 9로 표시
            img[j, i] = 9  # 특징점을 원본 영상에 9로 표시

# --- Part 3: 결과 출력 및 시각화 ---
# NumPy 배열 출력 시 소수점 이하 2자리까지만 표시하도록 설정
np.set_printoptions(precision=2)
print("dy (Iy):\n", dy)               # ① y방향 미분 결과 출력
print("\ndx (Ix):\n", dx)               # ② x방향 미분 결과 출력
print("\ndyy (Iy^2):\n", dyy)           # ③ I_y^2 결과 출력
print("\ndxx (Ix^2):\n", dxx)           # ④ I_x^2 결과 출력
print("\ndyx (Iy*Ix):\n", dyx)           # ⑤ I_y * I_x 결과 출력
print("\ngdyy (Smoothed Iy^2):\n", gdyy) # ⑥ 가우시안 적용된 I_y^2 결과 출력
print("\ngdxx (Smoothed Ix^2):\n", gdxx) # ⑦ 가우시안 적용된 I_x^2 결과 출력
print("\ngdyx (Smoothed Iy*Ix):\n", gdyx) # ⑧ 가우시안 적용된 I_y * I_x 결과 출력
print("\nC (Harris Response):\n", C)    # ⑨ 해리스 코너 응답(C) 맵 출력 # 특징 가능성 맵
print("\nImage with Corners:\n", img)   # ⑩ 특징점이 9로 표시된 최종 이미지 출력

# 해리스 응답(C) 맵을 확대하여 시각화
popping = np.zeros((160, 160), np.uint8)  # 화소 확인 가능하게 16배로 확대
for j in range(0, 160):
    for i in range(0, 160):
        # C 맵의 값을 [0, 255] 범위의 밝기 값으로 변환하여 popping 영상에 저장
        # 값 조정(0.06 더하기) 및 스케일링(700 곱하기)은 시각화를 위한 것임
        popping[j, i] = np.uint8(np.clip(((C[j // 16, i // 16] + 0.06) * 700), 0, 255)) # np.clip 추가

# 확대된 해리스 응답 맵(popping) 표시
cv.imshow('Image Display2 (Harris Response Scaled)', popping) # ⑪
cv.waitKey()
cv.destroyAllWindows()