import cv2 as cv
import mediapipe as mp

# MediaPipe 관련 모듈 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Pose 모델 초기화
# static_image_mode: True이면 이미지, False이면 비디오 스트림에 최적화
# enable_segmentation: True이면 사람 분할 마스크도 함께 생성
# min_detection_confidence: 최소 검출 신뢰도
# min_tracking_confidence: 최소 추적 신뢰도
pose = mp_pose.Pose(static_image_mode=False, enable_segmentation=True, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# 카메라 연결
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # BGR 이미지를 RGB로 변환 후 자세 추정 처리
    res = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    # 추정된 자세 랜드마크를 프레임에 그리기
    mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

    # 좌우 반전하여 거울처럼 보이게 한 후 출력
    cv.imshow('MediaPipe pose', cv.flip(frame, 1))

    if cv.waitKey(5) == ord('q'):
        # 'q' 키를 누르면 3D 랜드마크를 Matplotlib으로 시각화
        mp_drawing.plot_landmarks(res.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        break

cap.release()
cv.destroyAllWindows()