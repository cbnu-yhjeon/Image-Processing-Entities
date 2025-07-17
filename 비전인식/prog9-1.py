import numpy as np
import cv2 as cv
import sys


def construct_yolo_v3():
    # COCO 클래스 이름 파일 읽기
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]

    # YOLO v3 모델과 가중치 파일 로드
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = model.getLayerNames()
    # getUnconnectedOutLayers()가 2차원 배열로 반환될 수 있으므로 수정
    out_layers_indices = model.getUnconnectedOutLayers()
    if out_layers_indices.ndim == 2:  # e.g., [[200], [227], [254]]
        out_layers_indices = out_layers_indices.flatten()
    out_layers = [layer_names[i - 1] for i in out_layers_indices]

    return model, out_layers, class_names


def yolo_detect(img, yolo_model, out_layers):
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
model, out_layers, class_names = construct_yolo_v3()  # YOLO 모델 생성
colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔

img = cv.imread('soccer.jpg')
if img is None:
    sys.exit('파일이 없습니다.')

res = yolo_detect(img, model, out_layers)  # YOLO 모델로 물체 검출

for i in range(len(res)):  # 검출된 물체를 영상에 표시
    x1, y1, x2, y2, confidence, id = res[i]
    text = str(class_names[id]) + '%.3f' % confidence
    cv.rectangle(img, (x1, y1), (x2, y2), colors[id], 2)
    cv.putText(img, text, (x1, y1 + 30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)

cv.imshow("Object detection by YOLO v.3", img)

cv.waitKey()
cv.destroyAllWindows()