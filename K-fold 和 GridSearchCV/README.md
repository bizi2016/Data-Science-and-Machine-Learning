# K-fold 和 GridSearchCV

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

![img](100%20GPU.png)
