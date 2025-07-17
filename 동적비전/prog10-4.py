import cv2 as cv
import mediapipe as mp

# 이미지 읽기
img = cv.imread('BSDS_376001.jpg')

# MediaPipe 얼굴 검출 및 그리기 유틸리티 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 얼굴 검출 모델 설정
# model_selection=1: 2m 이내의 근거리 얼굴에 적합
# min_detection_confidence=0.5: 검출 신뢰도 임계값
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# BGR 이미지를 RGB로 변환하여 얼굴 검출 수행
res = face_detection.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

# 검출 결과 처리
if not res.detections:
    print('얼굴 검출에 실패했습니다. 다시 시도하세요.')
else:
    # 검출된 각 얼굴에 대해 경계 상자와 주요 지점 그리기
    for detection in res.detections:
        mp_drawing.draw_detection(img, detection)

    cv.imshow('Face detection by MediaPipe', img)

# 키 입력 대기 및 창 닫기
cv.waitKey()
cv.destroyAllWindows()