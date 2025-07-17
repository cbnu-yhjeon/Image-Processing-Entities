# 프로그램 4-7 GrabCut을 이용해 물체 분할하기

import cv2 as cv
import numpy as np

# --- Part 1: 초기 설정 및 페인팅 함수 정의 ---
img = cv.imread('soccer.jpg')  # 영상 읽기
img_show = np.copy(img)  # 붓칠을 디스플레이할 목적의 영상

mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
# 모든 화소를 배경일 것 같음(GC_PR_BGD)으로 초기화
mask[:, :] = cv.GC_PR_BGD

BrushSiz = 9  # 붓의 크기
LColor, RColor = (255, 0, 0), (0, 0, 255)  # 파란색(물체)과 빨간색(배경)

# 마우스 이벤트 콜백 함수 정의
def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)  # 왼쪽 버튼 클릭하면 파란색 (물체 확실)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)  # 마스크에 전경(GC_FGD) 표시
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)  # 오른쪽 버튼 클릭하면 빨간색 (배경 확실)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)  # 마스크에 배경(GC_BGD) 표시
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)  # 왼쪽 버튼 클릭하고 이동하면 파란색
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)  # 오른쪽 버튼 클릭하고 이동하면 빨간색

    cv.imshow('Painting', img_show) # 실시간으로 붓칠 결과 보여주기

# --- Part 2: 창 생성 및 마우스 콜백 연결 ---
cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

# --- Part 3: 사용자 입력 대기 및 GrabCut 실행 ---
while(True):  # 붓칠을 끝내려면 'q' 키를 누름
    # waitKey(1)로 변경하여 Painting 창이 실시간 업데이트 되도록 함
    if cv.waitKey(1) == ord('q'):
        break

# 여기부터 GrabCut 적용하는 코드
background = np.zeros((1, 65), np.float64)  # 배경 히스토그램 0으로 초기화
foreground = np.zeros((1, 65), np.float64)  # 물체 히스토그램 0으로 초기화

# GrabCut 실행 (반복 횟수: 5)
cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)

# 마스크에서 확실한 배경(GC_BGD)과 아마도 배경(GC_PR_BGD)인 영역은 0으로, 나머지는 1로 설정
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 원본 이미지에 마스크를 적용하여 결과 생성 (전경만 추출)
grab = img * mask2[:, :, np.newaxis]
cv.imshow('Grab cut image', grab)

cv.waitKey()
cv.destroyAllWindows()