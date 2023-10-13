####################
# pandas 读取csv
####################

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# 读取CSV文件
df = pd.read_csv('processed_data.csv')

# 将数据存储在data变量中（去除第一列标签）
data = df.iloc[:, 1:].values.astype(float)

# 将标签存储在label变量中（第一列）
label = df.iloc[:, 0].values.astype(float)


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
scatter = plt.scatter( data_2d[:, 0], data_2d[:, 1], c=label,
                       marker='.', alpha=0.5,
                       )
plt.title('PCA')
# plt.pause(0.1)

# auto legend
plt.legend( *scatter.legend_elements(prop='colors'),
            title='Class', loc='best' )

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
scatter = plt.scatter( data_2d[:, 0], data_2d[:, 1], c=label,
                       marker='.', alpha=0.5,
                       )
plt.title('t-SNE')
# plt.pause(0.1)

# auto legend
plt.legend( *scatter.legend_elements(prop='colors'),
            title='Class', loc='best' )

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
scatter = plt.scatter( data_2d[:, 0], data_2d[:, 1], c=label,
                       marker='.', alpha=0.5,
                       )
plt.title('UMAP')
# plt.show()

# auto legend
plt.legend( *scatter.legend_elements(prop='colors'),
            title='Class', loc='best' )

print('UMAP')

# 保存图形为PNG格式图片
plt.savefig('UMAP.png', dpi=200)
plt.close()

"""
data.shape = (48842, 92)
label.shape = (48842,)

PCA
t-SNE
中间一堆警告，因为版本更新迭代问题，希望以后能够修复吧
看着很烦，但是没啥后果
UMAP
"""



















