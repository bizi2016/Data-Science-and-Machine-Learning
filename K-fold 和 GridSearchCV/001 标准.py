import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

import matplotlib.pyplot as plt

####################
# 读取数据
####################

# 从processed_data.csv中读取数据
df = pd.read_csv('processed_data.csv')

# 将数据拆分为特征和标签
X = df.drop(columns=['class'])
y = df['class']

# 划分训练集和测试集，比例为4:1
X_train, X_test, y_train, y_test = \
         train_test_split( X, y, test_size=0.2, stratify=y )

####################
# xgb
####################

# 计算正类别的权重
scale_pos_weight = np.sum(y_train==0) / np.sum(y_train==1)
print('scale_pos_weight =', scale_pos_weight)
print()

# 使用XGBoost训练分类模型
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['error', 'logloss'],
    'scale_pos_weight': scale_pos_weight,
    'n_jobs': -1,
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# 在测试集上获取预测概率
y_proba = model.predict_proba(X_test)[:, 1]
y_pred =    y_proba >= 0.5

####################
# 测试结果
####################

# 计算准确率、F1和AUC
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# 计算ROC曲线的点
fpr, tpr, _ = roc_curve(y_test, y_proba)

####################
# 绘制曲线
####################

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print('acc =', acc)
print('f1 =', f1)
print('auc =', auc)

"""
scale_pos_weight = 3.1793774735265803

acc = 0.8367284266557478
f1 = 0.715534153736401
auc = 0.9305006401062574
"""
















