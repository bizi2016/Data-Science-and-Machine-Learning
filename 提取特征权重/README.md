# 提取特征权重（不适用K-fold）

## XGBoost 已加入GPU模式，完全释放性能

```python3
tree_method='gpu_hist',  # 使用GPU进行树的构建
predictor='gpu_predictor',  # 使用GPU进行预测
```

```python3
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
```

## 绘图
![img](feature_importance.png)

## csv
| index | feature |	importance |
| ---- | ---- | ---- |
| 16 | marital-status_1 | 0.43330792 |
| 3	| capital-gai | 0.05495588 |
| 2	| education-num	| 0.036196277 |
| 24 | occupation_2	| 0.030374518 |
| 4	| capital-loss | 0.021946002 |
| 34 | occupation_12	| 0.020106567 |
| 37 | relationship_0	| 0.020088488 |
| 30 | occupation_8	| 0.019657772 |
| 0	| age |	0.012189315 |
| 25 | occupation_3	| 0.011672632 |
