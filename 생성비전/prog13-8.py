import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
import matplotlib.pyplot as plt

# --- 1. 데이터 준비 및 하이퍼파라미터 설정 ---
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 부류 1(automobile) 추출
# y_train의 shape을 (50000, 1)에서 (50000,)으로 변경하여 isin 함수에 적용
x_train = x_train[np.isin(y_train.flatten(), [1])]
# [-1, 1] 정규화
x_train = (x_train.astype('float32') / 255.0) * 2.0 - 1.0
# 잠복 공간의 차원
zdim = 100


# --- 2. 모델(판별망 D, 생성망 G) 정의 ---

# 판별망 D 생성 함수 (CIFAR-10 이미지 크기 32x32x3에 맞게 수정)
def make_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=0.2), input_shape=in_shape))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2)))  # 16x16
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2)))  # 8x8
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2)))  # 4x4
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model


# 생성망 G 생성 함수 (CIFAR-10 이미지 생성에 맞게 수정)
def make_generator(zdim):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation=LeakyReLU(alpha=0.2), input_dim=zdim))
    model.add(Reshape((4, 4, 256)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2)))  # 8*8
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2)))  # 16*16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2)))  # 32*32
    model.add(Conv2D(3, (3, 3), padding='same', activation='tanh'))  # 32x32x3 컬러 이미지 생성
    return model


# --- 3. GAN 모델 및 헬퍼 함수 정의 (프로그램 13-6과 동일) ---

# GAN 모델 생성 함수
def make_gan(G, D):
    D.trainable = False
    model = Sequential()
    model.add(G)
    model.add(D)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model


# 진짜 샘플 묶음 생성 함수
def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    y = np.ones((n_samples, 1))
    return x, y


# 잠복 공간 점 생성 함수
def generate_latent_points(zdim, n_samples):
    return np.random.randn(n_samples, zdim)


# 가짜 샘플 생성 함수
def generate_fake_samples(G, zdim, n_samples):
    x_input = generate_latent_points(zdim, n_samples)
    x = G.predict(x_input)
    y = np.zeros((n_samples, 1))
    return x, y


# --- 4. GAN 학습 함수 정의 및 실행 ---

# GAN 학습 함수
def train(G, D, GAN, dataset, zdim, n_epochs=200, batch_siz=128, verbose=0):
    n_batch = int(dataset.shape[0] / batch_siz)

    for epoch in range(n_epochs):
        for b in range(n_batch):
            x_real, y_real = generate_real_samples(dataset, batch_siz // 2)
            d_loss1, _ = D.train_on_batch(x_real, y_real)

            x_fake, y_fake = generate_fake_samples(G, zdim, batch_siz // 2)
            d_loss2, _ = D.train_on_batch(x_fake, y_fake)

            x_gan = generate_latent_points(zdim, batch_siz)
            y_gan = np.ones((batch_siz, 1))
            g_loss = GAN.train_on_batch(x_gan, y_gan)

        if verbose == 1:
            print('>%d, D(real)=%.3f, D(fake)%.3f G%.3f' % (epoch + 1, d_loss1, d_loss2, g_loss))

        if (epoch + 1) % 10 == 0:
            x_fake, y_fake = generate_fake_samples(G, zdim, 12)
            plt.figure(figsize=(20, 2))
            plt.suptitle('epoch ' + str(epoch + 1))
            for k in range(12):
                plt.subplot(1, 12, k + 1)
                img = (x_fake[k] + 1) / 2.0
                plt.imshow(img);
                plt.xticks([]);
                plt.yticks([])
            plt.show()


# --- 5. 모델 생성 및 학습 실행 ---
D = make_discriminator((32, 32, 3))
G = make_generator(zdim)
GAN = make_gan(G, D)
train(G, D, GAN, x_train, zdim, verbose=1)