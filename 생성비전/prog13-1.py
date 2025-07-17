import numpy as np

# (키, 몸무게) 데이터
X = np.array([[169, 70], [172, 68], [175, 78], [163, 58], [180, 80], [159, 76], [158, 52], [173, 69], [180, 75], [155, 50], [187, 90], [170, 66]])

# 모델 학습 (평균과 공분산 계산)
m = np.mean(X, axis=0)      # 평균
cv = np.cov(X, rowvar=False) # 공분산

# 새로운 샘플 5개 생성
gen = np.random.multivariate_normal(m, cv, 5)

# 생성된 샘플 출력
print(gen)