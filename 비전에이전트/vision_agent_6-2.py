from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
import sys
import cv2 as cv


class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')
        self.setGeometry(200, 200, 500, 100)

        self.videoButton = QPushButton('비디오 켜기', self)
        self.captureButton = QPushButton('프레임 잡기', self)
        self.saveButton = QPushButton('프레임 저장', self)
        self.quitButton = QPushButton('나가기', self)

        self.videoButton.setGeometry(10, 10, 100, 30)
        self.captureButton.setGeometry(110, 10, 100, 30)
        self.saveButton.setGeometry(210, 10, 100, 30)
        self.quitButton.setGeometry(310, 10, 100, 30)

        self.videoButton.clicked.connect(self.videoFunction)
        self.captureButton.clicked.connect(self.captureFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        self.quitButton.clicked.connect(self.quitFunction)

        self.cap = None
        self.frame = None
        self.capturedFrame = None

        self.statusLabel = QLabel('대기 중...', self)
        self.statusLabel.setGeometry(10, 50, 300, 30)

    def videoFunction(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            self.statusLabel.setText("카메라를 열 수 없습니다.")
            self.cap = None
            return

        self.statusLabel.setText("비디오 재생 중... ('q'로 중지)")
        self.videoButton.setEnabled(False)
        self.quitButton.setEnabled(False)

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                self.statusLabel.setText("프레임 읽기 실패. 루프 종료.")
                break
            cv.imshow('video display', self.frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.statusLabel.setText("비디오가 중지되었습니다.")
                break

        cv.destroyWindow('video display')
        self.videoButton.setEnabled(True)
        self.quitButton.setEnabled(True)
        if self.cap is not None and self.cap.isOpened():  # 'q'로 나올때도 릴리즈
            self.cap.release()
            self.cap = None  # 명시적으로 None 처리

    def captureFunction(self):
        if self.frame is not None:
            self.capturedFrame = self.frame.copy()
            cv.imshow('Captured Frame', self.capturedFrame)
            self.statusLabel.setText("프레임 캡처 완료.")
        else:
            self.statusLabel.setText("비디오가 재생 중이지 않거나 프레임이 없습니다.")

    def saveFunction(self):
        if self.capturedFrame is not None:
            fname, _ = QFileDialog.getSaveFileName(self, '파일 저장', './', "Image Files (*.png *.jpg *.bmp)")
            if fname:
                cv.imwrite(fname, self.capturedFrame)
                self.statusLabel.setText(f"'{fname}'으로 저장됨.")
            else:
                self.statusLabel.setText("저장 취소됨.")
        else:
            self.statusLabel.setText("저장할 캡처된 프레임이 없습니다.")

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
    win = Video()
    win.show()
    sys.exit(app.exec_())