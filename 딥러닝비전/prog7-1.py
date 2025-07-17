import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

# MNIST 데이터셋 로드 및 시각화
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = ds.mnist.load_data()
print(x_train_mnist.shape, y_train_mnist.shape, x_test_mnist.shape, y_test_mnist.shape) # ①

plt.figure(figsize=(24, 3))
plt.suptitle('MNIST', fontsize=30)
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train_mnist[i], cmap='gray') # ②
    plt.xticks([])
    plt.yticks([])
    plt.title(str(y_train_mnist[i]), fontsize=30)
plt.show()

# CIFAR-10 데이터셋 로드 및 시각화
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = ds.cifar10.load_data()
print(x_train_cifar.shape, y_train_cifar.shape, x_test_cifar.shape, y_test_cifar.shape) # ③
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(24, 3))
plt.suptitle('CIFAR-10', fontsize=30)
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train_cifar[i]) # ④
    plt.xticks([])
    plt.yticks([])
    plt.title(class_names[y_train_cifar[i, 0]], fontsize=30)
plt.show()