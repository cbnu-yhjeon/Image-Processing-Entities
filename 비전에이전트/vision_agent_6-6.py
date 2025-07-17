import cv2 as cv
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QComboBox, QFileDialog


class SpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('사진 특수 효과')
        self.setGeometry(200, 200, 800, 200)  # 창 너비 조정

        self.pictureButton = QPushButton('사진 읽기', self)
        self.embossButton = QPushButton('엠보싱', self)
        self.cartoonButton = QPushButton('카툰', self)
        self.sketchButton = QPushButton('연필 스케치', self)
        self.oilButton = QPushButton('유화', self)
        self.saveButton = QPushButton('저장하기', self)
        self.pickCombo = QComboBox(self)
        self.pickCombo.addItems(['엠보싱', '카툰', '연필 스케치(명암)', '연필 스케치(컬러)', '유화'])
        self.quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        self.pictureButton.setGeometry(10, 10, 100, 30)
        self.embossButton.setGeometry(110, 10, 100, 30)
        self.cartoonButton.setGeometry(210, 10, 100, 30)
        self.sketchButton.setGeometry(310, 10, 100, 30)
        self.oilButton.setGeometry(410, 10, 100, 30)
        self.saveButton.setGeometry(510, 10, 100, 30)
        self.pickCombo.setGeometry(510, 40, 180, 30)  # 위치 조정
        self.quitButton.setGeometry(620, 10, 100, 30)  # 위치 조정
        self.label.setGeometry(10, 50, 500, 140)  # 높이 조정

        self.pictureButton.clicked.connect(self.pictureOpenFunction)
        self.embossButton.clicked.connect(self.embossFunction)
        self.cartoonButton.clicked.connect(self.cartoonFunction)
        self.sketchButton.clicked.connect(self.sketchFunction)
        self.oilButton.clicked.connect(self.oilFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        self.quitButton.clicked.connect(self.quitFunction)

        self.img = None
        self.emboss = None
        self.cartoon = None
        self.sketch_gray = None
        self.sketch_color = None
        self.oil = None

    def pictureOpenFunction(self):
        fname, _ = QFileDialog.getOpenFileName(self, '사진 읽기', './')
        if fname:
            self.img = cv.imread(fname)
            if self.img is None:
                self.label.setText('파일을 읽을 수 없습니다.')
                return
            cv.imshow('Original Image', self.img)  # 창 제목 변경
            self.label.setText("이미지를 로드했습니다. 원하는 효과 버튼을 누르세요.")
        else:
            self.label.setText("파일 열기가 취소되었습니다.")

    def embossFunction(self):
        if self.img is None:
            self.label.setText("먼저 사진 파일을 읽어오세요.")
            return
        femboss = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray16 = np.int16(gray)
        self.emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255))
        cv.imshow('Emboss', self.emboss)
        self.label.setText("엠보싱 효과 적용 완료.")

    def cartoonFunction(self):
        if self.img is None:
            self.label.setText("먼저 사진 파일을 읽어오세요.")
            return
        try:
            self.cartoon = cv.stylization(self.img, sigma_s=60, sigma_r=0.45)
            cv.imshow('Cartoon', self.cartoon)
            self.label.setText("카툰 효과 적용 완료.")
        except AttributeError:
            self.label.setText("오류: cv.stylization 사용 불가. opencv-contrib-python 필요?")
        except Exception as e:
            self.label.setText(f"카툰 효과 적용 중 오류: {e}")

    def sketchFunction(self):
        if self.img is None:
            self.label.setText("먼저 사진 파일을 읽어오세요.")
            return
        try:
            self.sketch_gray, self.sketch_color = cv.pencilSketch(self.img, sigma_s=60, sigma_r=0.07, shade_factor=0.02)
            cv.imshow('Pencil sketch(gray)', self.sketch_gray)
            cv.imshow('Pencil sketch(color)', self.sketch_color)
            self.label.setText("연필 스케치 효과 적용 완료.")
        except AttributeError:
            self.label.setText("오류: cv.pencilSketch 사용 불가. opencv-contrib-python 필요?")
        except Exception as e:
            self.label.setText(f"연필 스케치 적용 중 오류: {e}")

    def oilFunction(self):
        if self.img is None:
            self.label.setText("먼저 사진 파일을 읽어오세요.")
            return
        try:
            # cv.xphoto 네임스페이스 확인 (OpenCV 버전에 따라 다를 수 있음)
            if hasattr(cv, 'xphoto') and hasattr(cv.xphoto, 'oilPainting'):
                self.oil = cv.xphoto.oilPainting(self.img, 10, 1, cv.COLOR_BGR2Lab)  # COLOR_BGR2Lab 시도
            else:
                # oilPainting이 xphoto 모듈에 없을 경우, 기본 모듈에 있는지 확인 (구버전)
                # 또는 다른 방법 시도 (예: 직접 구현 또는 다른 라이브러리)
                # 여기서는 일단 cv.oilPainting 시도 (오류 발생 가능성 높음)
                self.oil = cv.oilPainting(self.img, 10, 1)  # 오래된 방식 시도 (오류날 수 있음)

            cv.imshow('Oil painting', self.oil)
            self.label.setText("유화 효과 적용 완료.")
        except AttributeError:
            self.label.setText("오류: cv.xphoto.oilPainting 또는 cv.oilPainting 사용 불가. OpenCV 버전 확인 필요.")
        except Exception as e:
            self.label.setText(f"유화 효과 적용 중 오류: {e}")

    def saveFunction(self):
        fname, _ = QFileDialog.getSaveFileName(self, '파일 저장', './', "Images (*.png *.jpg *.bmp)")
        if not fname:
            self.label.setText("저장 취소됨.")
            return

        i = self.pickCombo.currentIndex()
        img_to_save = None

        if i == 0 and self.emboss is not None:
            img_to_save = self.emboss
        elif i == 1 and self.cartoon is not None:
            img_to_save = self.cartoon
        elif i == 2 and self.sketch_gray is not None:
            img_to_save = self.sketch_gray
        elif i == 3 and self.sketch_color is not None:
            img_to_save = self.sketch_color
        elif i == 4 and self.oil is not None:
            img_to_save = self.oil

        if img_to_save is not None:
            cv.imwrite(fname, img_to_save)
            self.label.setText(f"'{fname}'으로 저장 완료.")
        else:
            selected_effect = self.pickCombo.currentText()
            self.label.setText(f"'{selected_effect}' 효과가 적용된 이미지가 없습니다. 먼저 효과를 적용하세요.")

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

    def closeEvent(self, event):
        cv.destroyAllWindows()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SpecialEffect()
    win.show()
    sys.exit(app.exec_())