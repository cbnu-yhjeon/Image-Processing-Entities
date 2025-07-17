import cv2 as cv
import mediapipe as mp
import numpy as np  # 알파 채널 처리를 위해 numpy 임포트

# 증강 현실에 쓸 장신구(주사위 이미지) 로드
# cv.IMREAD_UNCHANGED: 알파 채널(투명도)을 포함하여 이미지를 읽음
dice = cv.imread('dice.png', cv.IMREAD_UNCHANGED)
# 장신구 이미지 크기 조절
dice = cv.resize(dice, dsize=(0, 0), fx=0.1, fy=0.1)
w, h = dice.shape[1], dice.shape[0]

# MediaPipe 얼굴 검출 및 그리기 유틸리티 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 얼굴 검출 모델 설정
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# 카메라 연결
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # 얼굴 검출 수행
    res = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if res.detections:
        for det in res.detections:
            # 오른쪽 눈의 위치 정보 가져오기
            p = mp_face_detection.get_key_point(det, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

            # 눈 위치를 중심으로 장신구를 오버레이할 좌표 계산
            x1, x2 = int(p.x * frame.shape[1] - w // 2), int(p.x * frame.shape[1] + w // 2)
            y1, y2 = int(p.y * frame.shape[0] - h // 2), int(p.y * frame.shape[0] + h // 2)

            # 장신구가 프레임 경계 내에 있을 때만 처리
            if x1 > 0 and y1 > 0 and x2 < frame.shape[1] and y2 < frame.shape[0]:
                # 알파 블렌딩을 이용한 오버레이
                alpha = dice[:, :, 3:] / 255  # 투명도를 나타내는 알파값 (0~1)

                # 배경 = 원래 이미지 * (1-알파)
                # 객체 = 장신구 이미지 * 알파
                # 최종 이미지 = 배경 + 객체
                frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - alpha) + dice[:, :, :3] * alpha

    # 좌우 반전하여 거울처럼 보이게 한 후 출력
    cv.imshow('MediaPipe Face AR', cv.flip(frame, 1))
    if cv.waitKey(5) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()