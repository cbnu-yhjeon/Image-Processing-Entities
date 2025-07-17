import keras_cv
import matplotlib.pyplot as plt

# Stable Diffusion 모델 초기화 (이미지 크기 512x512)
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

# 텍스트 프롬프트를 사용하여 이미지 3개 생성
img = model.text_to_image('A cute rabbit in an avocado armchair', batch_size=3)

# 생성된 이미지들을 시각화
for i in range(len(img)):
    plt.subplot(1, len(img), i+1)
    plt.imshow(img[i])
    plt.axis('off')

plt.show()