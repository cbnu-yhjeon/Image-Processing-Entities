import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import backend as K

# --- 1. 데이터 준비 및 하이퍼파라미터 설정 ---
# MNIST를 읽고 신경망에 입력할 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# 잠복 공간의 차원
zdim = 32

# --- 2. VAE 모델 구축 ---

# 샘플링 함수 (재매개변수화 기법)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], zdim), mean=0.0, stddev=0.1)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# 인코더
encoder_input = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(encoder_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
x = Flatten()(x)
z_mean = Dense(zdim)(x)
z_log_var = Dense(zdim)(x)
encoder_output = Lambda(sampling)([z_mean, z_log_var])
# VAE의 인코더는 평균, 로그 분산, 샘플을 모두 출력
model_encoder = Model(encoder_input, [z_mean, z_log_var, encoder_output])

# 디코더
decoder_input = Input(shape=(zdim,))
x = Dense(3136)(decoder_input)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', strides=(1, 1))(x)
decoder_output = x
model_decoder = Model(decoder_input, decoder_output)

# 인코더와 디코더를 결합하여 VAE 모델 구축
model_input = encoder_input
model_output = model_decoder(encoder_output)
model = Model(model_input, model_output)

# VAE 손실 함수 정의 (복원 손실 + KL 손실)
reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(model_input, model_output), axis=(1, 2)))
kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
total_loss = reconstruction_loss + kl_loss
model.add_loss(total_loss)

# --- 3. 모델 학습 및 샘플 생성 ---
model.compile(optimizer='adam')
model.fit(x_train, x_train, epochs=50, batch_size=128, validation_data=(x_test, x_test))

# 테스트 집합에서 임의로 두 샘플 선택
i = np.random.randint(x_test.shape[0])
j = np.random.randint(x_test.shape[0])
x = np.array((x_test[i], x_test[j]))
# 두 샘플을 잠복 공간으로 매핑 (출력 중 세 번째 값인 샘플 z를 사용)
z = model_encoder.predict(x)[2]

# 두 샘플 사이에서 등간격 점을 만듦
zz = np.zeros((11, zdim))
alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(11):
    zz[i] = (1.0 - alpha[i]) * z[0] + alpha[i] * z[1]

# 등간격 점에서 가짜 샘플 생성
gen = model_decoder.predict(zz)

# --- 4. 생성된 샘플 시각화 ---
plt.figure(figsize=(20, 4))
for i in range(11):
    plt.subplot(1, 11, i + 1)
    plt.imshow(gen[i].reshape(28, 28), cmap='gray'); plt.xticks([]); plt.yticks([])
    plt.title(str(alpha[i]))
plt.show()