import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 데이터셋 구성
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 0 부류만 추출
X = x_train[np.isin(y_train, [0])]
# 이미지를 1차원 벡터로 변환
X = X.reshape((X.shape[0], 28*28))

# 모델 학습 (평균과 공분산 계산)
m = np.mean(X, axis=0)
cv = np.cov(X, rowvar=False)

# 샘플 생성 (5개)
gen = np.random.multivariate_normal(m, cv, 5)

# 샘플 그리기
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    # 1차원 벡터를 28x28 이미지로 재구성하여 출력
    plt.imshow(gen[i].reshape((28, 28)), cmap='gray'); plt.xticks([]); plt.yticks([])
plt.show()