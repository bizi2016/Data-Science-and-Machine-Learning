import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

from sklearn.svm import SVC



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



# 使用SVM进行训练
clf = SVC( # kernel='rbf',  # 默认 RBF函数
           kernel='linear',  # 线性 其实效果还不错
           probability=True,
           class_weight='balanced',
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
accuracy = 0.7837821069611615
precision = 0.5302832953435364
recall = 0.844438682914182
f1 = 0.6514651465146515
roc_auc = 0.8945067756166825

Confusion Matrix
====================
[[9376 2885]
 [ 600 3257]]
"""


























