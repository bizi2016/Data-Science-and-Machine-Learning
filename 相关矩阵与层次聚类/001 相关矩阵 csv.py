import pandas as pd

# 读取CSV文件
df = pd.read_csv('processed_data.csv')



def dump(method):
    
    # 计算皮尔森相关系数
    correlation_matrix = df.corr(method=method)

    # 将相关矩阵输出到CSV文件，设置index和header为True以保留行和列的标签
    correlation_matrix.to_csv( f'{method}.csv',
                               index=True, header=True )



methods = ['pearson', 'kendall', 'spearman']

for method in methods:
    dump(method)
