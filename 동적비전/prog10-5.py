import cv2 as cv
import mediapipe as mp

# MediaPipe 얼굴 검출 및 그리기 유틸리티 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 얼굴 검출 모델 설정
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# 웹캠 연결
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # BGR 프레임을 RGB로 변환하여 얼굴 검출 수행
    res = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    # 검출 결과가 있으면 프레임에 그리기
    if res.detections:
        for detection in res.detections:
            mp_drawing.draw_detection(frame, detection)

    # 좌우 반전하여 거울 모드로 출력
    cv.imshow('MediaPipe Face Detection from video', cv.flip(frame, 1))

    # 'q' 키를 누르면 종료
    if cv.waitKey(5) == ord('q'):
        break

# 자원 해제
cap.release()
cv.destroyAllWindows()