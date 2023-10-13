import pandas as pd

# 读取train.csv和test.csv文件
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 使用pd.concat()函数沿着行方向将两个数据框连接，并保留唯一的表头
merged_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# 将合并后的数据保存到一个新的CSV文件中
merged_data.to_csv('merged_data.csv', index=False)

# 验证合并后的数据
print('merged_data.head()')
print('='*20)
print(merged_data.head())

"""
merged_data.head()
====================
   age   workclass  fnlwgt  ... hours-per-week  native-country   class
0   25     Private  178478  ...             40   United-States   <=50K
1   23   State-gov   61743  ...             35   United-States   <=50K
2   46     Private  376789  ...             15   United-States   <=50K
3   55           ?  200235  ...             50   United-States    >50K
4   36     Private  224541  ...             40     El-Salvador   <=50K

[5 rows x 15 columns]
"""
