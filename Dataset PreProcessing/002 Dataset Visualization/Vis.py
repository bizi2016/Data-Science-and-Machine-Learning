import warnings
warnings.filterwarnings('ignore')

import gc
gc.enable()

####################
# 读取csv
####################

import numpy as np
import pandas as pd

# 读取带有表头的CSV文件
data = pd.read_csv('merged_data.csv', header=0)

# 打印数据的前几行，以验证是否成功读取
print('data.head()')
print('='*20)
print(data.head())
print()

####################
# 直方图可视化
####################

"""
`sns.distplot()` 是Seaborn库（一个基于Matplotlib的数据可视化库）中的一个函数，
用于绘制单变量分布的直方图和核密度估计。它将数据集的分布可视化，展示了数据的分布情况，
包括数据的中心趋势和离散度。

具体来说，`sns.distplot()` 的主要功能有：

1. 绘制直方图（Histogram）：
直方图是一种常见的展示数据分布的图形，它将数据划分成若干个区间（称为"bin"），
并统计每个区间内的数据点数量。直方图的高度表示每个区间内数据点的频数，
用以表示数据分布的形状。

2. 绘制核密度估计图（Kernel Density Estimation，KDE）：
KDE 是一种非参数化的估计方法，用于估计概率密度函数。
它通过在每个数据点周围放置一个核函数，并将它们加和起来，生成一条平滑的曲线，
代表数据的密度分布。KDE图可以帮助你更好地理解数据的分布情况，尤其是在直方图中的一些细节。
"""

import seaborn as sns
import matplotlib.pyplot as plt

"""
sns.distplot(x=data['age'], axlabel='Age', color='darkblue')

# 设置整个图的标题
plt.title('Distribution of Age')
plt.grid()

# plt.show()

plt.savefig('image/Age.png', dpi=200)
plt.close()
"""

def gen_hist(name):
    
    sns.distplot(x=data[ name[0] ], axlabel=name[0], color='darkblue')

    # 设置整个图的标题
    plt.title('Distribution of ' + name[1])
    plt.grid()
    plt.tight_layout()

    plt.savefig('image/distplot/' + name[1] + '.png', dpi=200)
    plt.close()


"""
names = [ ['age', 'Age'],
          ['fnlwgt', 'Final Weight'],
          ['education-num', 'Education Number'],
          ['capital-gain', 'Capital Gain'],
          ['capital-loss', 'Capital Loss'],
          ['hours-per-week', 'Numbers of Hours per Week'],  
    ]

for name in names:
    print(name)
    gen_hist( name )

# fnlwgt​: final weight
# 最终权重。人口普查认为该条目代表的人数。
"""

"""
sns.set(style='darkgrid')
sns.displot(x=data['class'])

plt.title('Distribution of Income')
plt.grid()
plt.tight_layout()

plt.savefig('image/Income.png', dpi=200)
plt.close()
"""


data['class'] = data['class'].apply(lambda x: x.replace('<=50K', '0'))
data['class'] = data['class'].apply(lambda x: x.replace('>50K', '1'))
data['class'] = data['class'].astype(int)

def gen_col(name):

    sns.set(style='darkgrid')
    sns.barplot(x=name[0], y='class', data=data)
    plt.xticks(rotation=90)

    plt.title('Distribution of ' + name[1])
    plt.grid()
    plt.tight_layout()

    # plt.show()

    plt.savefig('image/barplot/' + name[1] + '.png', dpi=200)
    plt.close()


"""
names = [ ['workclass', 'Workclass'],
          ['education', 'Education'],
          ['marital-status', 'Marital Status'],
          ['relationship', 'Relationship'],
          ['occupation', 'Occupation'],
          ['sex', 'Sex'],
          ['race', 'Race'],
          ['native-country', 'Native Country'],
    ]

for name in names:
    print(name)
    gen_col(name)
"""

"""
sns.set( style='darkgrid', font_scale=1.5 )
sns.catplot( x='sex', y='class', data=data, kind='bar', col='race',
             height=5, aspect=1 )

plt.title('Demographic Distribution')
plt.grid()
plt.tight_layout()

# plt.show()

plt.savefig('image/Demographic Distribution.png', dpi=200)
plt.close()
"""






























