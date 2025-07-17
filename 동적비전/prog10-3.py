import numpy as np
import cv2 as cv
import sys
from sort import Sort  # SORT 라이브러리 임포트


# ----------------- [프로그램 9-1]의 함수들 -----------------
def construct_yolo_v3():
    """
    YOLOv3 모델을 구성하고, 클래스 이름과 출력 레이어를 반환합니다.
    """
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = model.getLayerNames()
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
    test_img = cv.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True)
    yolo_model.setInput(test_img)
    output3 = yolo_model.forward(out_layers)

    box, conf, id = [], [], []
    for output in output3:
        for vec85 in output:
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
                w, h = int(vec85[2] * width), int(vec85[3] * height)
                x, y = int(centerx - w / 2), int(centery - h / 2)
                box.append([x, y, x + w, y + h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return objects


# ----------------- [프로그램 10-3] 메인 코드 -----------------
model, out_layers, class_names = construct_yolo_v3()  # YOLO 모델 생성
colors = np.random.uniform(0, 255, size=(100, 3))  # 100개 색으로 트랙 구분

# SORT 추적기 초기화
sort = Sort()

# 카메라 연결
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    # YOLO로 객체 검출
    res = yolo_detect(frame, model, out_layers)

    # 검출된 객체 중 사람(부류 0)만 필터링
    persons = [res[i] for i in range(len(res)) if res[i][5] == 0]

    # SORT 추적기 업데이트
    if len(persons) == 0:
        # 사람이 검출되지 않으면 빈 배열로 업데이트
        tracks = sort.update(np.empty((0, 6)))
    else:
        # 검출된 사람 정보를 넘파이 배열로 변환하여 업데이트
        tracks = sort.update(np.array(persons))

    # 추적 결과 시각화
    for i in range(len(tracks)):
        x1, y1, x2, y2, track_id = tracks[i].astype(int)
        cv.rectangle(frame, (x1, y1), (x2, y2), colors[track_id], 2)
        cv.putText(frame, str(track_id), (x1 + 10, y1 + 40), cv.FONT_HERSHEY_PLAIN, 3, colors[track_id], 2)

    cv.imshow('Person tracking by SORT', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()  # 카메라와 연결을 끊음
cv.destroyAllWindows()