import cv2 as cv
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog

class Orim(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('오림')
        self.setGeometry(200, 200, 700, 200)

        self.fileButton = QPushButton('파일', self)
        self.paintButton = QPushButton('페인팅', self)
        self.cutButton = QPushButton('오림', self)
        self.incButton = QPushButton('+', self)
        self.decButton = QPushButton('-', self)
        self.saveButton = QPushButton('저장', self)
        self.quitButton = QPushButton('나가기', self)

        self.fileButton.setGeometry(10, 10, 100, 30)
        self.paintButton.setGeometry(120, 10, 100, 30)
        self.cutButton.setGeometry(230, 10, 100, 30)
        self.incButton.setGeometry(340, 10, 50, 30)
        self.decButton.setGeometry(400, 10, 50, 30)
        self.saveButton.setGeometry(460, 10, 100, 30)
        self.quitButton.setGeometry(570, 10, 100, 30)

        self.statusLabel = QLabel('파일 버튼을 눌러 이미지를 여세요.', self)
        self.statusLabel.setGeometry(10, 50, 680, 30)

        self.img = None
        self.img_show = None
        self.mask = None
        self.grabImg = None

        self.BrushSiz = 5
        self.LColor = (255, 0, 0)
        self.RColor = (0, 0, 255)

        self.fileButton.clicked.connect(self.fileOpenFunction)
        self.paintButton.clicked.connect(self.paintFunction)
        self.cutButton.clicked.connect(self.cutFunction)
        self.incButton.clicked.connect(self.incFunction)
        self.decButton.clicked.connect(self.decFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        self.quitButton.clicked.connect(self.quitFunction)

    def fileOpenFunction(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', './')
        if fname:
            self.img = cv.imread(fname)
            if self.img is None:
                self.statusLabel.setText('파일을 열 수 없습니다.')
                return

            self.img_show = np.copy(self.img)
            cv.imshow('Painting', self.img_show)
            self.statusLabel.setText(f"'{fname}' 이미지가 로드되었습니다. 페인팅 모드를 활성화하세요.")

            self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
            self.mask[:, :] = cv.GC_PR_BGD

    def paintFunction(self):
        if self.img_show is not None:
            cv.setMouseCallback('Painting', self.painting)
            self.statusLabel.setText("페인팅 모드: LButton-전경(파랑), RButton-배경(빨강)")
        else:
            self.statusLabel.setText("먼저 파일을 열어주세요.")

    def painting(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)
        elif event == cv.EVENT_RBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)

        cv.imshow('Painting', self.img_show)

    def cutFunction(self):
        if self.img is None or self.mask is None:
            self.statusLabel.setText("파일을 먼저 열고 페인팅을 수행해주세요.")
            return
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv.grabCut(self.img, self.mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask == cv.GC_BGD) | (self.mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
        self.grabImg = self.img * mask2[:, :, np.newaxis]
        cv.imshow('Scissoring', self.grabImg)
        self.statusLabel.setText("GrabCut 실행 완료. 'Scissoring' 창에서 결과 확인.")

    def incFunction(self):
        self.BrushSiz = min(20, self.BrushSiz + 1)
        self.statusLabel.setText(f"브러시 크기: {self.BrushSiz}")

    def decFunction(self):
        self.BrushSiz = max(1, self.BrushSiz - 1)
        self.statusLabel.setText(f"브러시 크기: {self.BrushSiz}")

    def saveFunction(self):
        if self.grabImg is not None:
            fname, _ = QFileDialog.getSaveFileName(self, '파일 저장', './', "Image Files (*.png *.jpg *.bmp)")
            if fname:
                cv.imwrite(fname, self.grabImg)
                self.statusLabel.setText(f"오린 이미지가 '{fname}'으로 저장됨.")
            else:
                self.statusLabel.setText("저장 취소됨.")
        else:
            self.statusLabel.setText("저장할 오린 이미지가 없습니다. 먼저 '오림'을 실행하세요.")

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

    def closeEvent(self, event):
        cv.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Orim()
    win.show()
    sys.exit(app.exec_())