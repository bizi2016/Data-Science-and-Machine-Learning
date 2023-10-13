####################
# load mnist
####################

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.keras.datasets import fashion_mnist



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


data = np.concatenate( (x_train, x_test) )
data = np.reshape( data, (data.shape[0], data.shape[1]*data.shape[2]) )

label = np.concatenate( (y_train, y_test) )


print('data.shape =', data.shape)
print('label.shape =', label.shape)
print()



samples = 5000

####################
# pca
####################

import time
from sklearn.decomposition import PCA

start = time.time()  # timer

pca = PCA(n_components=3)
data_3d = pca.fit_transform(data[:samples])

end = time.time()  # timer



# define axes
ax = plt.axes(projection='3d')

scatter = ax.scatter(
    data_3d[:, 0],
    data_3d[:, 1],
    data_3d[:, 2],
    c=label[:samples], cmap='jet', s=5 ) # s=size, alpha=0.3

# auto legend
ax.legend( *scatter.legend_elements(prop='colors'),
           title='Class', loc='best' )

print('PCA', 'time =', end-start)

ax.grid()
plt.title('PCA')
ax.set_xlabel('X-axis', fontsize=15)
ax.set_ylabel('Y-axis', fontsize=15)
ax.set_zlabel('Z-axis', fontsize=15)

plt.tight_layout()
plt.show()

####################
# tsne
####################

from sklearn.manifold import TSNE

start = time.time()  # timer

tsne = TSNE( n_components=3,
             learning_rate='auto',
             init='random',
             n_jobs=-1,
             )
data_3d = tsne.fit_transform(data[:samples])

end = time.time()  # timer



# define axes
ax = plt.axes(projection='3d')

scatter = ax.scatter(
    data_3d[:, 0],
    data_3d[:, 1],
    data_3d[:, 2],
    c=label[:samples], cmap='jet', s=5 ) # s=size, alpha=0.3

# auto legend
ax.legend( *scatter.legend_elements(prop='colors'),
           title='Class', loc='best' )

print('t-SNE', 'time =', end-start)

ax.grid()
plt.title('t-SNE')
ax.set_xlabel('X-axis', fontsize=15)
ax.set_ylabel('Y-axis', fontsize=15)
ax.set_zlabel('Z-axis', fontsize=15)

plt.tight_layout()
plt.show()

####################
# umap
####################

# pip install umap-learn
from umap import UMAP

start = time.time()  # timer

umap = UMAP(n_components=3, n_jobs=-1)
data_3d = umap.fit_transform(data[:samples])

end = time.time()  # timer



# define axes
ax = plt.axes(projection='3d')

scatter = ax.scatter(
    data_3d[:, 0],
    data_3d[:, 1],
    data_3d[:, 2],
    c=label[:samples], cmap='jet', s=5 ) # s=size, alpha=0.3

# auto legend
ax.legend( *scatter.legend_elements(prop='colors'),
           title='Class', loc='best' )

print('UMAP', 'time =', end-start)

ax.grid()
plt.title('UMAP')
ax.set_xlabel('X-axis', fontsize=15)
ax.set_ylabel('Y-axis', fontsize=15)
ax.set_zlabel('Z-axis', fontsize=15)

plt.tight_layout()
plt.show()

"""
data.shape = (70000, 784)
label.shape = (70000,)

PCA time = 0.08001184463500977
t-SNE time = 13.3955078125
中间一堆警告，因为版本更新迭代问题，希望以后能够修复吧
看着很烦，但是没啥后果
UMAP time = 24.42484760284424
"""



















