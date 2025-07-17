# 프로그램 4-6 정규화 절단 알고리즘으로 영역 분할하기

import skimage
import numpy as np
import cv2 as cv
import time
# skimage.graph 모듈을 직접 임포트하거나 아래처럼 경로를 수정합니다.
# from skimage import graph # 이 줄을 추가하거나 아래처럼 수정

coffee = skimage.data.coffee()

start = time.time()
# start_label=1 옵션은 최신 버전에서 없어졌을 수 있으므로 제거합니다.
# skimage 0.19 버전부터 start_label 파라미터는 제거되었습니다.
# slic = skimage.segmentation.slic(coffee, compactness=20, n_segments=600, start_label=1)
slic = skimage.segmentation.slic(coffee, compactness=20, n_segments=600)

# 경로 수정: skimage.future.graph -> skimage.graph
g = skimage.graph.rag_mean_color(coffee, slic, mode='similarity')
ncut = skimage.graph.cut_normalized(slic, g) # 정규화 절단
print(coffee.shape, ' Coffee 영상을 분할하는 데 ', time.time()-start, '초 소요')

# mark_boundaries 함수는 레이블 이미지를 입력으로 받으므로 ncut 대신 slic 사용 고려
# ncut 결과는 보통 분할된 영역 레이블을 나타냅니다. mark_boundaries에는 이 레이블 맵을 넣어야 합니다.
# ncut 함수의 반환값을 확인하고, 그것이 레이블 맵이라면 그대로 사용합니다.
# 만약 아니라면, slic 결과를 사용해야 할 수도 있습니다. 우선 ncut을 사용해봅니다.
marking = skimage.segmentation.mark_boundaries(coffee, ncut)
ncut_coffee = np.uint8(marking * 255.0)

cv.imshow('Normalized cut', cv.cvtColor(ncut_coffee, cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()