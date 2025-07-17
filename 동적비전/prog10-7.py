import cv2 as cv
import mediapipe as mp

# MediaPipe 관련 모듈 초기화
mp_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# FaceMesh 모델 초기화
# max_num_faces: 최대 검출 얼굴 수
# refine_landmarks: 눈, 입술 주변 랜드마크까지 정교하게 검출할지 여부
# min_detection_confidence: 최소 검출 신뢰도
# min_tracking_confidence: 최소 추적 신뢰도
mesh = mp_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

# 카메라 연결
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # BGR 이미지를 RGB로 변환 후 FaceMesh 처리
    res = mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    # 검출된 얼굴 그물망이 있으면 그리기
    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks:
            # 1. 얼굴 전체의 삼각 분할(tessellation) 그리기
            mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks,
                                      connections=mp_mesh.FACEMESH_TESSELLATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_tessellation_style())

            # 2. 얼굴 윤곽선(contours) 그리기
            mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks,
                                      connections=mp_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())

            # 3. 눈동자(irises) 그리기
            mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks,
                                      connections=mp_mesh.FACEMESH_IRISES,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style())

    # 좌우 반전하여 거울처럼 보이게 한 후 출력
    cv.imshow('MediaPipe Face Mesh', cv.flip(frame, 1))

    if cv.waitKey(5) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()