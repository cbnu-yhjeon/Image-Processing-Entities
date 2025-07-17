import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 데이터셋 구성
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 0 부류만 추출
X = x_train[np.isin(y_train, [0])]
# 이미지를 1차원 벡터로 변환
X = X.reshape((X.shape[0], 28*28))

# GMM의 가우시안 개수 k를 8로 설정
k = 8

# GMM 모델 학습
gm = GaussianMixture(n_components=k).fit(X)

# GMM으로부터 새로운 샘플 10개 생성
# gm.sample()은 (샘플, 레이블) 튜플을 반환
gen = gm.sample(n_samples=10)

# 학습된 가우시안 8개의 평균을 그림
plt.figure(figsize=(20, 4))
plt.suptitle("Means of 8 Gaussian components")
for i in range(k):
    plt.subplot(1, 10, i+1)
    plt.imshow(gm.means_[i].reshape((28, 28)), cmap='gray'); plt.xticks([]); plt.yticks([])
plt.show()

# 생성된 샘플 10개를 그림
plt.figure(figsize=(20, 4))
plt.suptitle("10 generated samples")
for i in range(10):
    plt.subplot(1, 10, i+1)
    # gen[0]에 샘플 데이터가 들어있음
    plt.imshow(gen[0][i].reshape((28, 28)), cmap='gray'); plt.xticks([]); plt.yticks([])
plt.show()