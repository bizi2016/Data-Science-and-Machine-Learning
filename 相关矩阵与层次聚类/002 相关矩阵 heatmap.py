import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取原始数据的CSV文件
data = pd.read_csv('processed_data.csv')

# 使用Seaborn库创建热图
plt.figure( figsize=(24, 20), dpi=200 )

# annot=True, fmt='.2f'
sns.heatmap( data.corr(), annot=False, cmap='coolwarm',
             xticklabels=data.columns, yticklabels=data.columns )  
plt.title('Correlation Heatmap')

plt.tight_layout()
# plt.show()

# 保存热图为PNG图片，指定文件名为correlation_heatmap.png
plt.savefig( 'correlation_heatmap.png', dpi=200 )
plt.close()
