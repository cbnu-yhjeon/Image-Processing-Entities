import cv2 as cv
import numpy as np
import tensorflow as tf
import winsound
import pickle
import sys
from PyQt5.QtWidgets import *

# 사전 학습된 모델과 견종 이름 불러오기
cnn = tf.keras.models.load_model('cnn_for_stanford_dogs.h5')  # 모델 읽기
dog_species = pickle.load(open('dog_species_names.txt', 'rb'))  # 견종 이름


class DogSpeciesRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('견종 인식')
        self.setGeometry(200, 200, 700, 100)

        # 버튼 생성
        fileButton = QPushButton('강아지 사진 열기', self)
        recognitionButton = QPushButton('품종 인식', self)
        quitButton = QPushButton('나가기', self)

        # 버튼 위치 및 크기 설정
        fileButton.setGeometry(10, 10, 100, 30)
        recognitionButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)

        # 버튼 클릭 시 연결될 함수 지정
        fileButton.clicked.connect(self.pictureOpenFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

    def pictureOpenFunction(self):
        # 파일 대화상자를 통해 이미지 파일 열기
        fname = QFileDialog.getOpenFileName(self, '강아지 사진 읽기', './')
        self.img = cv.imread(fname[0])
        if self.img is None:
            sys.exit('파일을 찾을 수 없습니다.')

        cv.imshow('Dog image', self.img)

    def recognitionFunction(self):
        # 이미지를 모델 입력 형식에 맞게 변환
        x = np.reshape(cv.resize(self.img, (224, 224)), (1, 224, 224, 3))
        res = cnn.predict(x)[0]  # 예측
        top5 = np.argsort(-res)[:5]

        top5_dog_species_names = [dog_species[i] for i in top5]

        # 상위 5개 견종의 이름과 확률을 이미지에 표시
        for i in range(5):
            prob = '(' + str(res[top5[i]]) + ')'
            name = str(top5_dog_species_names[i]).split('-')[1]
            cv.putText(self.img, prob + name, (10, 100 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow('Dog image', self.img)
        winsound.Beep(1000, 500)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()


# PyQt 애플리케이션 실행
app = QApplication(sys.argv)
win = DogSpeciesRecognition()
win.show()
app.exec_()