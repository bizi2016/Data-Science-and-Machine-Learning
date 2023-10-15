import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

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
scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
print('scale_pos_weight =', scale_pos_weight)
print()

####################
# k-fold
####################

# 初始化K折交叉验证
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

####################
# 需要尝试的模型参数
####################
"""
# 测试环境：5950x，3200内存双通道，全部线程，不到1分钟
# 测试用：训练总量 1*5 = 5
param_grid = {
    'learning_rate': [0.1],
    'max_depth': [6],
    'min_child_weight': [1],
    'subsample': [1.0],
    'colsample_bytree': [1.0],
}
"""
"""
# 测试环境：5950x，3200内存双通道，全部线程，不到1小时
# S: 训练总量 243*5 = 1215
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}
"""
"""
# 测试环境：5950x，3200内存双通道，全部线程，不到3小时
# M: 训练总量 729*5 = 3645
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'n_estimators': [100, 200, 300]
}
"""

# 测试环境：5950x，3200内存双通道，全部线程，不到12小时，是本次例子使用的模型
# L: 训练总量 1296*5 = 6480
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'n_estimators': [100, 200, 300]
}

"""
# 至少需要超算部署，至少需要100个CPU
# AMD EPYC 7742
# 8 CPU，1024 线程 48小时以内（数据不同）
# XL: 训练总量 16384*5 = 81920
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2, 0.3],
    'n_estimators': [100, 150, 200, 250]
}
"""
"""
# 至少需要超算部署，至少需要100个CPU
# AMD EPYC 7742
# 16 CPU，2048 线程 120小时以内（数据不同）
# XXXL: 训练总量 1500000*5 = 7500000
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.1, 0.5, 1, 2],
    'reg_lambda': [0.1, 0.5, 1, 2, 5],
    'n_estimators': [100, 200, 300, 400, 500]
}
"""
"""
# 仅供参考修改参数使用
# 训练总量 8064000*5 = 40320000
param_grid = {
    # 学习率
    'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1, 0.2],  # 默认0.3
    # 树参数
    'n_estimators': [50, 100, 150, 200, 300],  # 多少棵树，默认100
    'max_depth': [3, 4, 5, 6, 7, 9, 12, 15, 17, 25],  # 默认6
    'min_child_weight': [1, 2, 3, 4, 5, 7],  # 默认1
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # 默认1
    'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 默认1
    'gamma': [0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # 默认0

    # reg_alpha: L1正则化项参数
    # reg_lambda: L2正则化项参数
    'reg_alpha': [0, 0.01, 0.1, 1.0],  # 默认0
    'reg_lambda': [0, 0.1, 0.5, 1.0],  # 默认1
}
"""
####################
# 分类器初始化
####################

# 初始化XGBoost分类器
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
)

####################
# 暴力搜索
####################

# 使用GridSearchCV进行参数搜索
# f1 accuracy
grid_search = GridSearchCV( estimator=xgb_model, param_grid=param_grid, 
                            scoring='roc_auc', cv=k_fold, n_jobs=1,
                            verbose=1,  # 模型多开这里不多开，SVM多开
                            )
"""
verbose=0: 什么都不输出，没有任何信息显示
verbose=1: 显示进度条，同时显示每个交叉验证折叠的进度
verbose=2: 显示每个交叉验证折叠的进度和详细信息，包括每个参数组合的搜索进度
verbose=3: 显示每个交叉验证折叠的进度和更详细的信息，包括每个参数组合的搜索进度和性能指标
"""

import time
start_time = time.time()

# 执行网格搜索
grid_result = grid_search.fit(X, y)

end_time = time.time()
elapsed_time = end_time - start_time
print()
print(f'程序执行时间：{elapsed_time:.2f}秒')

# 输出最优参数和得分
print()
print('最优参数')
print('='*20)
print(grid_result.best_params_)
print()

print('最优AUC')
print('='*20)
print(grid_result.best_score_)
print()

####################
# 运行最好的模型
####################

# 绘制ROC曲线
plt.figure(figsize=(8, 6))

# 初始化结果列表
acc_list = []
f1_list = []
auc_list = []

for i, (train_indices, test_indices) in enumerate(k_fold.split(X, y)):
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    model = grid_result.best_estimator_.fit(X_train, y_train)

    # 在测试集上获取预测概率
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred =     y_proba >= 0.5

    # 计算准确率、F1和AUC
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    acc_list.append(acc)
    f1_list.append(f1)
    auc_list.append(auc)

    ####################
    # 画ROC曲线
    ####################
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot( fpr, tpr, lw=2, alpha=0.5,
              label=f'Fold {i+1} AUC={auc:.2f}' )

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

####################
# 三分评估
####################

# 计算平均准确率、F1和AUC
mean_acc = np.mean(acc_list)
mean_f1 = np.mean(f1_list)
mean_auc = np.mean(auc_list)

print('mean_acc =', mean_acc)
print('mean_f1 =', mean_f1)
print('mean_auc =', mean_auc)

####################
# 保存模型
####################

import joblib

# 获取最佳模型
best_model = grid_result.best_estimator_

# 保存最佳模型到文件
joblib.dump(best_model, 'best_model.pkl')

# 加载最佳模型
# loaded_model = joblib.load('best_model.pkl')

"""
scale_pos_weight = 3.179173440574998

Fitting 5 folds for each of 1296 candidates, totalling 6480 fits

程序执行时间：13488.77秒

最优参数
====================
{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 3, 'n_estimators': 300, 'subsample': 0.9}

最优AUC
====================
0.9305210671928135

mean_acc = 0.8366978115852106
mean_f1 = 0.7170647902030296
mean_auc = 0.9305210671928135
"""










