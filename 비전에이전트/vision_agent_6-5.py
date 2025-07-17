from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
import cv2 as cv
import numpy as np
import winsound
import sys


class Panorama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('파노라마 영상')
        self.setGeometry(200, 200, 700, 200)

        self.collectButton = QPushButton('영상 수집', self)
        self.showButton = QPushButton('영상 보기', self)
        self.stitchButton = QPushButton('봉합', self)
        self.saveButton = QPushButton('저장', self)
        self.quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        self.collectButton.setGeometry(10, 25, 100, 30)
        self.showButton.setGeometry(110, 25, 100, 30)
        self.stitchButton.setGeometry(210, 25, 100, 30)
        self.saveButton.setGeometry(310, 25, 100, 30)
        self.quitButton.setGeometry(450, 25, 100, 30)  # 위치 조정
        self.label.setGeometry(10, 70, 600, 100)  # 높이 및 너비 조정

        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)

        self.collectButton.clicked.connect(self.collectFunction)
        self.showButton.clicked.connect(self.showFunction)
        self.stitchButton.clicked.connect(self.stitchFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        self.quitButton.clicked.connect(self.quitFunction)

        self.cap = None
        self.imgs = []
        self.img_stitched = None

    def collectFunction(self):
        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        self.label.setText('c를 여러 번 눌러 수집하고 끝나면 q를 눌러 비디오를 끕니다.')

        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            self.label.setText('카메라 연결 실패')
            self.cap = None
            return

        self.imgs = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.label.setText("프레임 읽기 실패. 중단합니다.")
                break

            cv.imshow('video display', frame)
            key = cv.waitKey(1)

            if key == ord('c'):
                self.imgs.append(frame.copy())  # 프레임 복사본 저장
                self.statusLabel.setText(f'{len(self.imgs)} 개의 영상 수집됨.')  # statusLabel이 없으므로 label 사용
                self.label.setText(f'{len(self.imgs)} 개의 영상 수집됨. 계속 수집하거나 q로 종료하세요.')
            elif key == ord('q'):
                if self.cap is not None:  # cap이 None이 아닐때만 release
                    self.cap.release()
                    self.cap = None
                cv.destroyWindow('video display')
                break

        if len(self.imgs) >= 2:
            self.showButton.setEnabled(True)
            self.stitchButton.setEnabled(True)
            self.saveButton.setEnabled(True)
            self.label.setText(f'{len(self.imgs)} 개의 영상 수집 완료. 보기/봉합/저장 가능.')
        else:
            self.label.setText('최소 2개 이상의 영상이 필요합니다. 다시 수집하세요.')
            self.imgs = []  # 영상 리스트 초기화

    def showFunction(self):
        if len(self.imgs) < 1:
            self.label.setText('먼저 영상을 수집하세요.')
            return

        self.label.setText('수집된 영상은 ' + str(len(self.imgs)) + '장입니다.')

        # 이미지 크기 조절 비율 (예: 0.25)
        fx_resize, fy_resize = 0.25, 0.25

        # 첫 번째 이미지 리사이즈
        stack = cv.resize(self.imgs[0], dsize=(0, 0), fx=fx_resize, fy=fy_resize)

        # 나머지 이미지 리사이즈 및 수평 연결
        for i in range(1, len(self.imgs)):
            img_resized = cv.resize(self.imgs[i], dsize=(0, 0), fx=fx_resize, fy=fy_resize)
            # 세로 크기가 다르면 맞춰주기 (첫번째 이미지 기준)
            if stack.shape[0] != img_resized.shape[0]:
                new_width = int(img_resized.shape[1] * (stack.shape[0] / img_resized.shape[0]))
                img_resized = cv.resize(img_resized, (new_width, stack.shape[0]))

            stack = np.hstack((stack, img_resized))

        cv.imshow('Image collection', stack)

    def stitchFunction(self):
        if len(self.imgs) < 2:
            self.label.setText('봉합을 위해 최소 2개의 영상이 필요합니다.')
            return

        stitcher = cv.Stitcher_create()
        status, self.img_stitched = stitcher.stitch(self.imgs)

        if status == cv.Stitcher_OK:
            cv.imshow('Image stitched panorama', self.img_stitched)
            self.label.setText('파노라마 영상 생성 성공! 저장 가능합니다.')
            self.saveButton.setEnabled(True)  # 저장 버튼 활성화 (stitch 성공 시)
        else:
            try:
                winsound.Beep(3000, 500)
            except RuntimeError:
                print("경고: winsound는 Windows에서만 작동합니다.")
            except Exception as e:
                print(f"경고: 소리 재생 중 오류 - {e}")
            self.label.setText('파노라마 제작에 실패했습니다. 다시 시도하세요.')
            self.img_stitched = None  # 실패 시 초기화

    def saveFunction(self):
        if self.img_stitched is not None:
            fname, _ = QFileDialog.getSaveFileName(self, '파일 저장', './', "Image Files (*.png *.jpg *.bmp)")
            if fname:
                cv.imwrite(fname, self.img_stitched)
                self.label.setText(f"파노라마 영상이 '{fname}'으로 저장됨.")
            else:
                self.label.setText("저장 취소됨.")
        else:
            self.label.setText("저장할 파노라마 영상이 없습니다. 먼저 '봉합'을 실행하세요.")

    def quitFunction(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()
        self.close()

    def closeEvent(self, event):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Panorama()
    win.show()
    sys.exit(app.exec_())