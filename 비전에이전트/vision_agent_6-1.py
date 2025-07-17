
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel  # 필요한 위젯만 명시적으로 임포트
import sys
import winsound  # Windows에서만 작동


class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('삑 소리 내기')  # 원도우 이름 설정 (이미지 라인 08 유사)
        self.setGeometry(200, 200, 500, 100)  # x, y, width, height (이미지 라인 09 유사)

        # 버튼 생성 (이미지 라인 11-13)
        self.shortBeepButton = QPushButton('짧게 삑', self)
        self.longBeepButton = QPushButton('길게 삑', self)
        self.quitButton = QPushButton('나가기', self)

        # 레이블 생성 (이미지 라인 14)
        self.label = QLabel('환영합니다!', self)

        # 버튼 위치와 크기 지정 (이미지 라인 16-18)
        self.shortBeepButton.setGeometry(10, 10, 100, 30)
        self.longBeepButton.setGeometry(110, 10, 100, 30)
        self.quitButton.setGeometry(210, 10, 100, 30)

        # 레이블 위치와 크기 지정 (이미지 라인 19 유사, 약간 조정)
        self.label.setGeometry(10, 50, 480, 30)  # 버튼 아래로, 텍스트 잘 보이도록 너비 조정

        # 콜백 함수 연결 (이미지 라인 21-23)
        self.shortBeepButton.clicked.connect(self.shortBeepFunction)
        self.longBeepButton.clicked.connect(self.longBeepFunction)
        self.quitButton.clicked.connect(self.quitFunction)

    # 24 (이미지 시작 라인)
    # 25: def shortBeepFunction(self):
    def shortBeepFunction(self):
        # 26: self.label.setText('주파수 1000으로 0.5초 동안 삑 소리를 냅니다.')
        self.label.setText('주파수 1000으로 0.5초 동안 삑 소리를 냅니다.')
        try:
            # 27: winsound.Beep(1000,500)
            winsound.Beep(1000, 500)  # 주파수 1000Hz, 0.5초(500ms)간 소리
        except RuntimeError:
            self.label.setText('winsound.Beep 에러: Windows 환경에서만 소리가 납니다.')
        except Exception as e:
            self.label.setText(f'소리 재생 중 에러 발생: {e}')

    # 28
    # 29: def longBeepFunction(self):
    def longBeepFunction(self):
        # 30: self.label.setText('주파수 1000으로 3초 동안 삑 소리를 냅니다.')
        self.label.setText('주파수 1000으로 3초 동안 삑 소리를 냅니다.')
        try:
            # 31: winsound.Beep(1000,3000)
            winsound.Beep(1000, 3000)  # 주파수 1000Hz, 3초(3000ms)간 소리
        except RuntimeError:
            self.label.setText('winsound.Beep 에러: Windows 환경에서만 소리가 납니다.')
        except Exception as e:
            self.label.setText(f'소리 재생 중 에러 발생: {e}')

    # 32
    # 33: def quitFunction(self):
    def quitFunction(self):
        # 34: self.close()
        self.close()  # 현재 창을 닫습니다. QApplication.instance().quit()는 전체 애플리케이션 종료.
        # 이미지에서는 self.close()를 사용했으므로 그대로 반영합니다.


# 35 (이미지 시작 라인)
# 36: app=QApplication(sys.argv)
if __name__ == '__main__':  # Python 스크립트 실행 시 표준적으로 사용되는 구문
    app = QApplication(sys.argv)
    # 37: win=BeepSound()
    win = BeepSound()
    # 38: win.show()
    win.show()
    # 39: app.exec_()
    sys.exit(app.exec_())  # sys.exit()로 감싸는 것이 좀 더 표준적인 종료 방식입니다.