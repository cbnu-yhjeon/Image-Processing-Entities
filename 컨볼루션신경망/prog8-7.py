import pathlib
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory

# 데이터 경로 설정
data_path = pathlib.Path('datasets/stanford_dogs/images/Images')

# 훈련 데이터셋과 테스트 데이터셋 로드
train_ds = image_dataset_from_directory(data_path, validation_split=0.2, subset='training', seed=123, image_size=(224, 224), batch_size=16)
test_ds = image_dataset_from_directory(data_path, validation_split=0.2, subset='validation', seed=123, image_size=(224, 224), batch_size=16)

# DenseNet121 베이스 모델 설정
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Sequential 모델 구성
cnn = Sequential()
cnn.add(Rescaling(1.0/255.0))
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.75))
cnn.add(Dense(units=120, activation='softmax'))

# 모델 컴파일
cnn.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.000001), metrics=['accuracy'])

# 모델 학습
hist = cnn.fit(train_ds, epochs=200, validation_data=test_ds, verbose=2)

# 모델 평가 및 정확도 출력
print('정확률=', cnn.evaluate(test_ds, verbose=0)[1] * 100)

# 모델 저장
cnn.save('cnn_for_stanford_dogs.h5')  # 미세 조정된 모델을 파일에 저장

# 클래스 이름 저장
f = open('dog_species_names.txt', 'wb')
pickle.dump(train_ds.class_names, f)
f.close()

# 정확도 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()

# 손실 그래프
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()