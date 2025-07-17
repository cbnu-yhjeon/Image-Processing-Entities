import cv2 as cv
import mediapipe as mp

# MediaPipe 관련 모듈 초기화
mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Hands 모델 초기화
# max_num_hands: 최대 검출 손의 개수
# static_image_mode: True이면 이미지, False이면 비디오 스트림에 최적화
# min_detection_confidence: 최소 검출 신뢰도
# min_tracking_confidence: 최소 추적 신뢰도
hand = mp_hand.Hands(max_num_hands=2, static_image_mode=False, min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)

# 카메라 연결
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # BGR 이미지를 RGB로 변환 후 손 랜드마크 처리
    res = hand.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    # 검출된 손 랜드마크가 있으면 그리기
    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hand.HAND_CONNECTIONS,
                                      mp_styles.get_default_hand_landmarks_style(),
                                      mp_styles.get_default_hand_connections_style())

    # 좌우 반전하여 거울처럼 보이게 한 후 출력
    cv.imshow('MediaPipe Hands', cv.flip(frame, 1))

    if cv.waitKey(5) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()