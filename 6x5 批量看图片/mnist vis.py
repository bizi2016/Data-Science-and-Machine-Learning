####################
# 读取数据集
####################

import numpy as np

mnist = np.load('mnist.npz')
print('mnist.files =', mnist.files)
print()



x_train = mnist['x_train']
y_train = mnist['y_train']

x_test = mnist['x_test']
y_test = mnist['y_test']

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)
print()

print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)
print()

####################
# 显示图片
####################

import matplotlib.pyplot as plt

# 定义要展示的行数和列数
rows, cols = 5, 6
num_images = rows * cols

# 创建一个新的图像
plt.figure()

# 遍历并展示图片
for i in range(num_images):
    
    plt.subplot( rows, cols, i+1 )  # plt 从1开始的
    plt.imshow(x_train[i], cmap='gray')  # 灰度图像
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')  # 不显示坐标轴

plt.tight_layout()
plt.show()













