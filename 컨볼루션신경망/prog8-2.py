import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 데이터 준비 (CIFAR-10)
(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()

# CIFAR-10 이미지는 이미 (샘플수, 32, 32, 3) 형태로 로드됩니다.
# 별도의 reshape 과정 없이 정규화 및 원-핫 인코딩을 진행합니다.
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# CNN 모델 구축
cnn = Sequential()
# 블록 1
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # padding 기본값 'valid'
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))
# 블록 2
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))
# FC 블록
cnn.add(Flatten())
cnn.add(Dense(units=512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(units=10, activation='softmax'))

# 모델 컴파일
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 모델 학습
hist = cnn.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), verbose=2) # ①

# 모델 평가
res = cnn.evaluate(x_test, y_test, verbose=0)
print('정확률=', res[1] * 100) # ②

# 정확도 그래프
plt.plot(hist.history['accuracy']) # ③
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()

# 손실 그래프
plt.plot(hist.history['loss']) # ④
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()