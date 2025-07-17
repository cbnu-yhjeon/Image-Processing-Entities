import numpy as np
import cv2 as cv
import sys
import time  # 시간 측정을 위해 time 모듈 추가


def construct_yolo_v3():
    """
    YOLOv3 모델을 구성하고, 클래스 이름과 출력 레이어를 반환합니다.
    """
    # COCO 클래스 이름 파일 읽기
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]

    # YOLO v3 모델과 가중치 파일 로드
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = model.getLayerNames()

    # getUnconnectedOutLayers()가 버전에 따라 다른 형태로 반환될 수 있어 호환성 확보
    try:
        out_layers_indices = model.getUnconnectedOutLayers().flatten()
    except AttributeError:
        out_layers_indices = model.getUnconnectedOutLayers()

    out_layers = [layer_names[i - 1] for i in out_layers_indices]

    return model, out_layers, class_names


def yolo_detect(img, yolo_model, out_layers):
    """
    주어진 이미지에서 YOLO 모델을 사용하여 객체를 탐지합니다.
    """
    height, width = img.shape[0], img.shape[1]
    # 이미지를 blob으로 변환하여 네트워크 입력으로 준비
    test_img = cv.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True)

    yolo_model.setInput(test_img)
    output3 = yolo_model.forward(out_layers)

    box, conf, id = [], [], []  # 박스, 신뢰도, 부류 번호

    for output in output3:
        for vec85 in output:
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # 신뢰도가 50% 이상인 경우만 취함
                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
                w, h = int(vec85[2] * width), int(vec85[3] * height)
                x, y = int(centerx - w / 2), int(centery - h / 2)
                box.append([x, y, x + w, y + h])
                conf.append(float(confidence))
                id.append(class_id)

    # 비최대 억제(NMS) 적용
    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return objects


# ----------------- 메인 프로그램 -----------------
# [프로그램 9-1]의 모델 생성 부분
model, out_layers, class_names = construct_yolo_v3()
colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔

# [프로그램 9-3]의 비디오 처리 및 속도 측정 부분
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

start = time.time()
n_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    res = yolo_detect(frame, model, out_layers)

    for i in range(len(res)):
        x1, y1, x2, y2, confidence, id = res[i]
        text = str(class_names[id]) + '%.3f' % confidence
        cv.rectangle(frame, (x1, y1), (x2, y2), colors[id], 2)
        cv.putText(frame, text, (x1, y1 + 30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)

    cv.imshow("Object detection from video by YOLO v.3", frame)
    n_frame += 1

    key = cv.waitKey(1)
    if key == ord('q'):
        break

end = time.time()
print('처리한 프레임 수=', n_frame, ', 경과 시간=', end - start, '\n초당 프레임 수=', n_frame / (end - start))

cap.release()  # 카메라와 연결을 끊음
cv.destroyAllWindows()
