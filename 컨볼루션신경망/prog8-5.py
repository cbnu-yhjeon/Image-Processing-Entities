import tensorflow.keras.datasets as ds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
x_train = x_train.astype('float32'); x_train /= 255
x_train = x_train[0:15,]; y_train = y_train[0:15,] # 앞 15개에 대해서만 증대 적용
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 원본 이미지를 출력
plt.figure(figsize=(20, 2))
plt.suptitle("First 15 images in the train set")
for i in range(15):
    plt.subplot(1, 15, i+1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[int(y_train[i])])
plt.show()

# 이미지 증대기 생성
batch_size = 4  # 한 번에 생성하는 양(미니 배치)
generator = ImageDataGenerator(rotation_range=20.0, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
gen = generator.flow(x_train, y_train, batch_size=batch_size)

# 증대된 이미지를 3번 출력
for a in range(3):
    img, label = gen.next()  # 미니 배치만큼 생성
    plt.figure(figsize=(8, 2.4))
    plt.suptitle("Generator trial " + str(a+1))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(img[i])
        plt.xticks([]); plt.yticks([])
        plt.title(class_names[int(label[i])])
    plt.show()