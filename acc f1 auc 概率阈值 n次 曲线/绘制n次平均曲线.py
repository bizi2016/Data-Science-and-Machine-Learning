import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import matplotlib.pyplot as plt



names = [ 'processed_data',
          ]

methods = [ 'XGBoost',
            'ExtraTrees',
            'RandomForest',
            # 'SVM',
            ]



for name in names:

    print(name)  # 显示进度
    print()

    # 读取CSV文件
    data = pd.read_csv( 'csv/' + name + '.csv' )

    # 提取标签和特征
    y = data.iloc[:, 0]
    x = data.iloc[:, 1:]
    
    for method in methods:

        print(method)  # 显示进度

        # 用于保存每一种方法的 所有n次 | 每一个阈值 的所有结果
        acc_list_n = []
        f1_list_n = []
        auc_list_n = []

        # 对于不太好的数据集，做n次取平均会比较好
        for time in range(20):

            print(time, end=' ')  # 显示进度

            # 按照标签进行分层分割，设置训练集和测试集的分割比例为2:1
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.1, stratify=y,  # test_size=0.33,
                )

            

            if method == 'XGBoost':
                
                import xgboost as xgb

                # 计算scale_pos_weight
                scale_pos_weight = np.sum(y_train==0) / np.sum(y_train==1)
                
                # 初始化XGBoost分类器，并设置scale_pos_weight参数
                clf = xgb.XGBClassifier( n_jobs=-1, scale_pos_weight=scale_pos_weight )

            if method == 'ExtraTrees':

                from sklearn.ensemble import ExtraTreesClassifier
                # 初始化Extra Trees分类器
                clf = ExtraTreesClassifier( n_jobs=-1, class_weight='balanced' )
                
            if method == 'RandomForest':

                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier( n_jobs=-1, class_weight='balanced' )

            if method == 'SVM':

                from sklearn.svm import SVC
                clf = SVC( kernel='rbf',
                           class_weight='balanced',
                           probability=True,
                           )



            clf.fit( x_train, y_train )  # 训练模型
            y_pred = clf.predict_proba(x_test)[:, 1]  # 预测



            # 不同阈值的结果列表
            acc_list = []
            f1_list = []
            auc_list = []

            x_coord = np.linspace(0, 1, num=1000)

            for th in x_coord:

                # 大于等于th的是1，小于th的是0
                y_th =    y_pred >= th

                # 结果
                acc = accuracy_score(y_test, y_th)
                f1 = f1_score(y_test, y_th)
                auc = roc_auc_score(y_test, y_th)

                # 不同阈值的结果列表
                acc_list.append(acc)
                f1_list.append(f1)
                auc_list.append(auc)



            # 用于保存每一种方法的 所有n次 | 每一个阈值 的所有结果
            acc_list_n.append(acc_list)
            f1_list_n.append(f1_list)
            auc_list_n.append(auc_list)

        print()
        print()



        keys = [ { 'name': 'Accuracy', 'data': acc_list_n },
                 { 'name': 'F1', 'data': f1_list_n },
                 { 'name': 'AUC', 'data': auc_list_n },
                 ]

        for key in keys:

            # 绘制函数图像，并设置dpi为200
            plt.figure(figsize=(8, 6), dpi=200)

            for i in range(len(key['data'])):
                
                plt.plot( x_coord, key['data'][i],
                          color='gray', alpha=0.5,
                          )

            mean_data = np.mean(key['data'], axis=0)

            plt.plot( x_coord, mean_data,
                      color='blue', label='Average',
                      )



            # 找到最大值对应坐标
            max_x = x_coord[np.argmax(mean_data)]
            max_y = mean_data[np.argmax(mean_data)]

            # 标注最大值点的坐标
            plt.scatter( max_x, max_y, color='red', label=f'Max Point: ({max_x:.2f}, {max_y:.2f})' )
            '''
            plt.annotate( f'({max_x:.2f}, {max_y:.2f})', xy=(max_x, max_y), xytext=(-20, 20),
                          textcoords='offset points', arrowprops=dict(arrowstyle="->"),
                          )
            '''



            plt.xlabel('Threshold')
            plt.ylabel( key['name'] )
            plt.title( name + ' @ ' + method + '\n' +
                       key['name'] + ' threshold curve' )

            plt.legend()
            plt.grid(True)
            # plt.show()

            # 保存图像，并设置dpi为200
            plt.savefig( 'pic/' + key['name'] + '/' +
                         name + ' @ ' + method + '.png', dpi=200 )
            plt.close()

    print()



















