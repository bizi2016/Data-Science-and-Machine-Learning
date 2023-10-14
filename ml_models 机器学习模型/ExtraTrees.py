import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

from sklearn.ensemble import ExtraTreesClassifier



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



# 使用Extra Trees进行训练
clf = ExtraTreesClassifier( n_estimators=100,  # 默认 100
                            # criterion='gini',  # 默认 使用基尼指数作为分裂标准
                            criterion='entropy',  # 使用熵作为分裂标准
                            class_weight='balanced',
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
accuracy = 0.8383174091078297
precision = 0.6850044365572315
recall = 0.6004666839512575
f1 = 0.6399557888919591
roc_auc = 0.8840577816215234

Confusion Matrix
====================
[[11196  1065]
 [ 1541  2316]]
"""




















