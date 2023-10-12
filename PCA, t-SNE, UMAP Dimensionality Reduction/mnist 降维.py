####################
# load mnist
####################

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


data = np.concatenate( (x_train, x_test) )
data = np.reshape( data, (data.shape[0], data.shape[1]*data.shape[2]) )

label = np.concatenate( (y_train, y_test) )


print('data.shape =', data.shape)
print('label.shape =', label.shape)
print()

####################
# pca
####################

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

plt.figure( dpi=200 )
plt.scatter( data_2d[:, 0], data_2d[:, 1], c=label,
             cmap='jet', marker='.', alpha=0.1,
             )
plt.title('PCA')
# plt.pause(0.1)

print('PCA')

# 保存图形为PNG格式图片
plt.savefig('PCA.png', dpi=200)
plt.close()

####################
# tsne
####################

from sklearn.manifold import TSNE

tsne = TSNE( n_components=2,
             learning_rate='auto',
             init='random',
             n_jobs=-1,
             )
data_2d = tsne.fit_transform(data)

plt.figure( dpi=200 )
plt.scatter( data_2d[:, 0], data_2d[:, 1], c=label,
             cmap='jet', marker='.', alpha=0.1,
             )
plt.title('t-SNE')
# plt.pause(0.1)

print('t-SNE')

# 保存图形为PNG格式图片
plt.savefig('t-SNE.png', dpi=200)
plt.close()

####################
# umap
####################

# pip install umap-learn
from umap import UMAP

umap = UMAP(n_components=2, n_jobs=-1)
data_2d = umap.fit_transform(data)

plt.figure( dpi=200 )
plt.scatter( data_2d[:, 0], data_2d[:, 1], c=label,
             cmap='jet', marker='.', alpha=0.1,
             )
plt.title('UMAP')
# plt.show()

print('UMAP')

# 保存图形为PNG格式图片
plt.savefig('UMAP.png', dpi=200)
plt.close()

"""
data.shape = (70000, 784)
label.shape = (70000,)

PCA
t-SNE
中间一堆警告，因为版本更新迭代问题，希望以后能够修复吧
看着很烦，但是没啥后果
UMAP
"""



















