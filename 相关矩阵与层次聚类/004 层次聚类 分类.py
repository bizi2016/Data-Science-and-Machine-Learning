import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram



# 读取处理过的数据集的CSV文件
df = pd.read_csv('processed_data.csv')

df = df.head(1000)  # 只取前1000行数据

# 提取特征数据，假设你的特征列从第1列开始（索引为0）
features = df.iloc[:, 1:]



# 计算层次聚类
linkage_matrix = linkage(features.T, method='ward')  # 使用ward方法进行层次聚类



# 指定希望得到的簇的数量
num_clusters = 5

# 使用fcluster函数将数据分配到指定数量的簇中
cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# 获取每个特征的列名
column_names = list(features.columns)

# 打印每个样本所属的簇号以及簇的列名称
for i, column_name in enumerate(column_names):
    print(f'index={i}, {{ {column_name}: {cluster_labels[i]} }}')



# 计算阈值（可根据需要调整）
th = 50

# 绘制树状图
plt.figure(figsize=(12, 8))

dendrogram( linkage_matrix, labels=column_names,
            orientation='top', distance_sort='descending',
            show_leaf_counts=True,
            )

plt.axhline( y=th,
             color='r', linestyle='--', linewidth=2,
             label=f'Threshold = {th}',
             )

plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Feature Index')
plt.ylabel('Distance')

plt.legend()
plt.tight_layout()
plt.show()



























