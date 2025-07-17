import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import winsound # Windows 전용 모듈

model = tf.keras.models.load_model('dmlp_trained.h5')

img = None

def reset():
    global img
    img = np.ones((200, 520, 3), dtype=np.uint8) * 255
    for i in range(5):
        cv.rectangle(img, (10 + i * 100, 50), (10 + (i + 1) * 100, 150), (0, 0, 255)) # 빨간색 사각형
    cv.putText(img, 'e:erase s:show r:recognition q:quit', (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1) # 파란색 글씨

def grab_numerals():
    numerals = []
    for i in range(5):
        # 다음은 원본 이미지의 ROI 추출 논리를 설명하는 주석입니다.
        # 숫자 영역은 높이 98px (51부터 149미만), 너비 98px로 추출됩니다.
        # 각 사각형의 x 시작점은 10+i*100 이고, 안쪽 1픽셀부터 시작 (11+i*100).
        # BGR 중 B 채널(0번 인덱스)만 사용하여 2D 배열로 만듭니다.
        roi = img[51:149, 11 + i * 100 : 11 + i * 100 + 98, 0] # B 채널, 98x98 크기
        # roi=img[51:149,11+i*100:9+(i+1)*100,0] # 원본 이미지에 있었던 코드 라인 (참고용)
        roi = 255 - cv.resize(roi, (28, 28), interpolation=cv.INTER_CUBIC) # MNIST 형식(흰색 글씨, 검은 배경)으로 변환 및 리사이즈
        numerals.append(roi)
    numerals = np.array(numerals)
    return numerals

def show():
    numerals = grab_numerals()
    plt.figure(figsize=(25, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(numerals[i], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

def recognition():
    numerals = grab_numerals()
    numerals = numerals.reshape(5, 784)
    numerals = numerals.astype(np.float32) / 255.0
    res = model.predict(numerals) # 신경망 모델로 예측
    class_id = np.argmax(res, axis=1)
    for i in range(5):
        cv.putText(img, str(class_id[i]), (50 + i * 100, 180), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1) # 파란색 글씨 (인식 결과)
    try:
        winsound.Beep(1000, 500)
    except RuntimeError: # winsound가 지원되지 않는 환경 (예: Linux, macOS 또는 Windows 소리 장치 문제)
        print("Beep sound not available on this system.")
    except ImportError: # winsound 모듈 자체가 없는 경우
        print("winsound module not found. Beep sound is skipped.")

BrushSiz = 4
LColor = (0, 0, 0) # 검은색 (그리기 색상)

def writing(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), BrushSiz, LColor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), BrushSiz, LColor, -1)

reset()
cv.namedWindow('Writing')
cv.setMouseCallback('Writing', writing)

while True:
    cv.imshow('Writing', img)
    key = cv.waitKey(1)
    if key == ord('e'):
        reset()
    elif key == ord('s'):
        show()
    elif key == ord('r'):
        recognition()
    elif key == ord('q'):
        break

cv.destroyAllWindows()