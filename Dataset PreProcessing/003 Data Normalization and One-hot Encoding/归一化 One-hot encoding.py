####################
# 读取数据
####################

import numpy as np
import pandas as pd

# 读取带有表头的CSV文件
data = pd.read_csv('merged_data.csv')
'''
# 打印数据的前几行，以验证是否成功读取
print(data.head())
print()
'''
####################
# 删除列
####################

# 删除'education'列
data = data.drop(columns=['education'])

####################
# 归一化
####################

# 需要归一化的列：
# age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week

# 要进行最大最小归一化的列
col_to_norm = [
    'age', 'fnlwgt', 'education-num', 'capital-gain',
    'capital-loss', 'hours-per-week',
    ]

# 使用最大最小归一化
for column in col_to_norm:
    
    data_min = data[column].min()
    data_max = data[column].max()
    
    data[column] = (data[column] - data_min) / \
                   (data_max - data_min)
"""
# 使用Z-score进行每列的归一化
for column in col_to_norm:
    
    data_mean = data[column].mean()  # 计算平均值
    data_std = data[column].std()    # 计算标准差
    data[column] = (data[column] - data_mean) / data_std  # Z-score归一化
"""
'''
# 打印归一化后的数据的前几行
print(data.head())
print()
'''
####################
# one-hot encoding
####################

# 需要独热编码的列：
# workclass, marital-status, occupation,
# relationship, race, sex, native-country

# 要进行独热编码的列
col_to_encode = [
    'workclass', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country',
    ]

# 生成映射表
mapping_tables = {}

# 对每一列进行统计、映射为数字并进行独热编码
for column in col_to_encode:

    unique_values = data[column].unique()
    mapping = { value: index for index, value in enumerate(unique_values) }
    mapping_tables[column] = mapping
    
    # 按映射表映射为数字
    column_mapping = data[column].map(mapping)
    
    # 进行独热编码
    column_encoded = pd.get_dummies(column_mapping, prefix=column)
    
    # 将独热编码的结果与原始数据合并
    data = pd.concat([data, column_encoded], axis=1)
    
    # 删除原始的列
    data = data.drop(columns=[column])
'''
# 打印映射表
print(mapping_tables)
print()

# 打印独热编码后的数据的前几行
print(data.head())
print()
'''
####################
# label
####################

# 因为label有特殊符号，所以开启了保护
# 使用strip()函数去除字符串两边的空格，然后使用map() / replace()函数将'<=50K'替换为0，'>50K'替换为1
data['class'] = data['class'].str.strip().map({'<=50K': 0, '>50K': 1})



# 将'class'列从数据框中移除并保存到变量class_column中
class_column = data.pop('class')

# 在数据框的第一列位置插入class_column列
data.insert(0, 'class', class_column)


'''
# 打印处理后的数据的前几行
print(data.head())
'''
####################
# 保存
####################

# csv
data.to_csv('processed_data.csv', index=False)



# npz

# 将DataFrame转换为NumPy数组
data_array = data.to_numpy()

# 将数据保存为压缩过的npz文件
np.savez_compressed('processed_data.npz', data=data)



# 保存映射表 mapping_tables

# 将字典保存为txt文件
with open('mapping_tables.txt', 'w') as fw:
    for key, value in mapping_tables.items():
        print(key, file=fw)
        print(value, file=fw)
        print(file=fw)
        


####################
# 读取npz
####################

data_new = np.load('processed_data.npz')
print('data_new.files =', data_new.files)
print()

array_new = data_new['data']
print('array_new.shape =', array_new.shape)

"""
data_new.files = ['data']

array_new.shape = (48842, 93)
"""







































