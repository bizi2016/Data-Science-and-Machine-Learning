import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

import xgboost as xgb



# 读取CSV文件
df = pd.read_csv('processed_data.csv')

# 提取特征和标签
X = df.drop(columns=['class'])
y = df['class']

# 分割数据集，按照label的比例来分割
X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                     test_size=0.33,
                                                     stratify=y,
                                                     )



# 计算正负样本的数量比例
pos_num = df['class'].sum()
neg_num = len(df) - pos_num
scale_pos_weight = neg_num / pos_num
# print('scale_pos_weight =', scale_pos_weight)

# 使用XGBoost进行训练，设置scale_pos_weight参数
clf = xgb.XGBClassifier( scale_pos_weight=scale_pos_weight,  # 设置正负样本权重比例
                         n_estimators=100,  # 默认 100
                         learning_rate=0.1,  # 默认0.1
                         n_jobs=-1,
                         )

clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 获取预测概率
y_prob = clf.predict_proba(X_test)[:, 1]



# 计算各种评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# 计算ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)



# 绘制ROC曲线
plt.figure( dpi=200 )
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



# 打印结果
print('accuracy =', accuracy)
print('precision =', precision)
print('recall =', recall)
print('f1 =', f1)
print('roc_auc =', roc_auc)
print()

print('Confusion Matrix')
print('=' * 20)
print(conf_matrix)

"""
accuracy = 0.8322992927162179
precision = 0.6037396619920892
recall = 0.8706248379569613
f1 = 0.713026860600913
roc_auc = 0.9296303476475923

Confusion Matrix
====================
[[10057  2204]
 [  499  3358]]
"""


























