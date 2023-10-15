import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


"""
# 读取原始数据的CSV文件
df = pd.read_csv('iris.csv')

# 使用replace方法将字符串标签替换为数字
mapping = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
df['variety'] = df['variety'].replace(mapping)
"""
# 读取原始数据的CSV文件
df = pd.read_csv('processed_data.csv')

# 转置，树状图显示的是行
df = df.T

# 计算层次聚类
linkage_matrix = linkage(df, method='ward')  # 使用ward方法进行层次聚类

# 绘制树状图
plt.figure( figsize=(10, 10), dpi=300 )

dendrogram( linkage_matrix, labels=df.index,
            orientation='top', distance_sort='descending',
            show_leaf_counts=True,
            )

plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Sample Index')
plt.ylabel('Distance')

plt.tight_layout()
# plt.show()

# 保存图片，设置dpi为200
plt.savefig( 'processed_data.png', dpi=300 )
plt.close()
