import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
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

# 计算正类别的权重
temp = np.sum(y==0) / np.sum(y==1)
print('scale_pos_weight =', temp)
print()

####################
# k_fold
####################

# 初始化K折交叉验证
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

####################
# 平均列表
####################

# 初始化结果列表
acc_list = []
f1_list = []
auc_list = []

fpr_list, tpr_list = [], []

####################
# k_fold
####################

# 进行K折交叉验证
# indices = index的复数
for train_indices, test_indices in k_fold.split(X, y):
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # 计算正类别的权重
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

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
    y_pred =     y_proba >= 0.5

    ####################
    # 测试分数
    ####################

    # 计算准确率、F1和AUC
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 计算ROC曲线的点
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    acc_list.append(acc)
    f1_list.append(f1)
    auc_list.append(auc)

    fpr_list.append(fpr); tpr_list.append(tpr)

####################
# 算平均
####################

# 计算平均准确率、F1和AUC
mean_acc = np.mean(acc_list)
mean_f1 = np.mean(f1_list)
mean_auc = np.mean(auc_list)

print('mean_acc =', mean_acc)
print('mean_f1 =', mean_f1)
print('mean_auc =', mean_auc)

####################
# 画图
####################

# 绘制ROC曲线
plt.figure(figsize=(8, 6))

for i in range(len(fpr_list)):
    plt.plot( fpr_list[i], tpr_list[i],
              lw=2, alpha=0.5, label=f'AUC = {auc_list[i]:.2f}' )

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

"""
scale_pos_weight = 3.179173440574998

mean_acc = 0.8377420061906703
mean_f1 = 0.7160756651174837
mean_auc = 0.9286348251381289
"""




























