import cv2 as cv
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
import winsound


class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('교통약자 보호')
        self.setGeometry(200, 200, 700, 200)

        self.signButton = QPushButton('표지판 등록', self)
        self.roadButton = QPushButton('도로 영상 불러옴', self)
        self.recognitionButton = QPushButton('인식', self)
        self.quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        self.signButton.setGeometry(10, 10, 100, 30)
        self.roadButton.setGeometry(120, 10, 140, 30)
        self.recognitionButton.setGeometry(270, 10, 100, 30)
        self.quitButton.setGeometry(380, 10, 100, 30)
        self.label.setGeometry(10, 50, 680, 30)

        self.signButton.clicked.connect(self.signFunction)
        self.roadButton.clicked.connect(self.roadFunction)
        self.recognitionButton.clicked.connect(self.recognitionFunction)
        self.quitButton.clicked.connect(self.quitFunction)

        self.signFiles = [['child.png', '어린이'], ['elder.png', '노인'], ['disabled.png', '장애인']]
        self.signImgs = []
        self.roadImg = None
        self.sift = None

    def signFunction(self):
        self.label.clear()
        self.label.setText('교통약자 표지판을 등록합니다.')
        self.signImgs = []
        loaded_count = 0
        for fname, name in self.signFiles:
            img = cv.imread(fname)
            if img is not None:
                self.signImgs.append(img)
                cv.imshow(f"Sign: {name} ({fname})", self.signImgs[-1])
                loaded_count += 1
            else:
                print(f"경고: 표지판 파일 '{fname}'을(를) 로드할 수 없습니다.")

        if loaded_count == len(self.signFiles):
            self.label.setText(f"{loaded_count}개의 모든 표지판 이미지가 등록되었습니다.")
        elif loaded_count > 0:
            self.label.setText(f"{loaded_count}개의 표지판 이미지가 등록되었습니다. 일부 파일 로드 실패.")
        else:
            self.label.setText("표지판 이미지를 하나도 등록하지 못했습니다. 파일 경로를 확인하세요.")

    def roadFunction(self):
        if not self.signImgs:
            self.label.setText('먼저 표지판을 등록하세요.')
            return
        else:
            fname, _ = QFileDialog.getOpenFileName(self, '파일 읽기', './')
            if fname:
                self.roadImg = cv.imread(fname)
                if self.roadImg is None:
                    self.label.setText('도로 영상을 찾을 수 없습니다.')
                    return

                cv.imshow('Road scene', self.roadImg)
                self.label.setText(f"'{fname}' 도로 영상이 로드되었습니다. 인식 버튼을 누르세요.")
            else:
                self.label.setText("도로 영상 불러오기가 취소되었습니다.")

    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText('먼저 도로 영상을 입력하세요.')
            return
        else:
            try:
                if self.sift is None:
                    self.sift = cv.SIFT_create()
            except AttributeError:
                self.label.setText("오류: SIFT_create()를 사용할 수 없습니다. opencv-contrib-python 설치가 필요할 수 있습니다.")
                return
            except Exception as e:
                self.label.setText(f"오류: SIFT 생성 중 에러 발생 - {e}")
                return

            KD = []
            for img in self.signImgs:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                kp, des = self.sift.detectAndCompute(gray, None)
                if des is not None:
                    KD.append((kp, des))
                else:
                    KD.append(([], None))

            grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
            road_kp, road_des = self.sift.detectAndCompute(grayRoad, None)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv.FlannBasedMatcher(index_params, search_params)

            GM = []
            for i, (sign_kp, sign_des) in enumerate(KD):
                good_match = []
                if sign_des is not None and road_des is not None and len(sign_kp) > 1 and len(road_kp) > 1:
                    try:
                        knn_match = matcher.knnMatch(sign_des, road_des, k=2)
                        T = 0.7
                        for m, n in knn_match:
                            if m.distance < T * n.distance:
                                good_match.append(m)
                    except cv.error as e:
                        print(f"Warning: knnMatch failed for sign {i} - {e}")
                GM.append(good_match)

            if not GM or not any(GM):
                self.label.setText("인식 실패: 도로 영상에서 표지판 특징점을 찾지 못했습니다.")
                return

            best = GM.index(max(GM, key=len))

            best_sign_kp, _ = KD[best]
            best_matches = GM[best]

            if best_matches:
                img_match = cv.drawMatches(self.signImgs[best], best_sign_kp, self.roadImg, road_kp, best_matches, None,
                                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv.imshow(f'Matches with {self.signFiles[best][1]} Sign', img_match)
            else:
                print(
                    f"Warning: No good matches found for the best sign candidate ({self.signFiles[best][1]}). Skipping match drawing.")

            sign_name = self.signFiles[best][1]
            self.label.setText(f'{sign_name} 보호구역입니다. 30km로 서행하세요.')

            try:
                winsound.Beep(3000, 500)
            except RuntimeError:
                print("경고: winsound.Beep은 Windows에서만 작동합니다.")
            except Exception as e:
                print(f"경고: 소리 재생 중 오류 발생 - {e}")

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

    def closeEvent(self, event):
        cv.destroyAllWindows()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TrafficWeak()
    win.show()
    sys.exit(app.exec_())