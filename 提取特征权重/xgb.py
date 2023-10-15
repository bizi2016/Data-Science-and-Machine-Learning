import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

import matplotlib.pyplot as plt

####################
# 读取
####################

# 读取CSV文件
df = pd.read_csv('processed_data.csv')

# 分割特征和标签
X = df.drop(columns=['class'])
y = df['class']

# 计算正类别权重
scale_pos_weight = np.sum(y==0) / np.sum(y==1)
print('scale_pos_weight =', scale_pos_weight)
print()

# 数据集分割
X_train, X_test, y_train, y_test = \
         train_test_split( X, y, test_size=0.33, stratify=y )

####################
# xgb
####################

# 定义XGBoost GPU模型参数
params = {
    'objective': 'binary:logistic',
    # 'mlogloss': Multiclass logloss
    # 'auc': Receiver Operating Characteristic Area under the Curve
    'eval_metric': 'logloss',
    
    'scale_pos_weight': scale_pos_weight,

    'tree_method': 'gpu_hist',  # 使用GPU进行树的构建
    'predictor': 'gpu_predictor',  # 使用GPU进行预测
    
    'n_jobs': -1,  # 一个CPU填不满显卡
}

# 定义XGBoost模型
gpu_clf = xgb.XGBClassifier(**params)

# 训练模型
gpu_clf.fit(X_train, y_train)

####################
# 预测
####################

# 预测
y_probs = gpu_clf.predict_proba(X_test)[:, 1]
y_pred =    y_probs >= 0.5

# 计算准确率
acc = accuracy_score(y_test, y_pred)

# 计算F1分数
f1 = f1_score(y_test, y_pred)

# 计算ROC AUC值
roc_auc = roc_auc_score(y_test, y_probs)

# 打印性能指标
print('acc =', acc)
print('f1 =', f1)
print('roc_auc =', roc_auc)
print()

"""
scale_pos_weight = 3.179173440574998

acc = 0.8353393721305373
f1 = 0.7123347062649036
roc_auc = 0.9257830354173192
"""

####################
# 绘制曲线
####################
"""
# 计算ROC曲线的点
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))

plt.plot( fpr, tpr, color='darkorange',
          label='ROC curve (area = {:.2f})'.format(roc_auc) )

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# plt.show()
# 保存图像
plt.savefig('ROC_curve.png', dpi=200)
plt.close()
"""
####################
# 提取特征重要性csv
####################

fi = gpu_clf.feature_importances_

# 创建包含特征名的DataFrame
feature_names = X.columns  # X是训练集
fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})

# 按照重要性从大到小排序
fi_df = fi_df.sort_values(by='importance', ascending=False)

# 将DataFrame保存为CSV文件
fi_df.to_csv('feature_importance.csv', index_label='index', index=True)

####################
# 提取特征重要性图
####################
"""
# 绘制特征重要性图
fig, ax = plt.subplots(figsize=(12, 16), dpi=200)
xgb.plot_importance( gpu_clf, ax=ax,
                     importance_type='weight', xlabel='Weight',
                     # max_num_features=10,  # 取最大10个
                     )

plt.tight_layout()

# plt.show()
# 保存图像
plt.savefig('feature_importance.png', dpi=200)
plt.close()
"""


























