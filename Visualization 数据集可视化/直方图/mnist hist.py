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
# 单一图片
####################

import matplotlib.pyplot as plt

index = 391

plt.subplot(121)

plt.imshow(x_train[index])
plt.axis('off')
plt.title(f'x_train[{index}]')



# 排除0，不然什么都看不到了
data = x_train[index].flatten()
data = data[data != 0]
data = data[data < 250]

plt.subplot(122)

# 绘制第392张图片的直方图
plt.hist( data, bins=256//10, range=(0, 256),
          density=True, color='blue', edgecolor='black', alpha=0.7,
          )
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title(f'Histogram of x_train[{index}]')

plt.tight_layout()
plt.show()

####################
# 整个数据集
####################

merged_images = np.concatenate( (x_train, x_test), axis=0 )
merged_labels = np.concatenate( (y_train, y_test), axis=0 )

# 排除0，不然什么都看不到了
data = merged_images.flatten()
data = data[data != 0]
data = data[data < 250]

# 绘制全部图片的直方图
plt.hist( data, bins=256//10, range=(0, 256),
          density=True, color='blue', edgecolor='black', alpha=0.7,
          )
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of mnist')

plt.tight_layout()
plt.show()

####################
# 分类显示
####################

# 获取所有标签的唯一值
unique_labels = np.unique(merged_labels)

# 遍历每个标签，生成直方图并放置在相应的子图中
for i, label in enumerate(unique_labels):

    plt.subplot(2, 5, i+1)
    
    # 选择当前标签对应的所有图片
    class_data = merged_images[ merged_labels == label ]
    class_data = class_data.flatten()
    class_data = class_data[class_data != 0]
    class_data = class_data[class_data < 250]

    # 绘制直方图
    plt.hist( class_data, bins=256//10, range=(0, 256),
              density=True, color='blue', edgecolor='black', alpha=0.7,
              )
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(label)

# 调整子图的间距和布局
plt.tight_layout()
plt.show()

























