####################
# 加载模型
####################

import joblib

# 获取最佳模型
# best_model = grid_result.best_estimator_

# 保存最佳模型到文件
# joblib.dump(best_model, 'best_model.pkl')

# 加载最佳模型
loaded_model = joblib.load('best_model.pkl')

####################
# 测试加载的模型
####################

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
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

####################
# k-fold
####################

# 初始化K折交叉验证
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

    # 在测试集上获取预测概率
    y_proba = loaded_model.predict_proba(X_test)[:, 1]
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

"""
"""

















