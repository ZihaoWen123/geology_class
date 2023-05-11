import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from data_process_train import data_process
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.inspection import permutation_importance

import joblib
import mlflow


def train(data_file, sheet_name, k_fold, miss_ratio, n_neighbors, chose_featrue, normal_method, impute_method, upsample_method, usehyparams_search, feat_importance_method, top_k):
    if sheet_name == 'Igneous Rocks Database':
        state = 'magma'
    else:
        state = 'deposit'

    x_train, x_test, y_train, y_test, keys = data_process(
        data_file=data_file,
        sheet_name=sheet_name,
        miss_ratio=miss_ratio,
        n_neighbors=n_neighbors,
        chose_featrue=chose_featrue,
        is_deep=False,
        normal_method=normal_method,
        impute_method=impute_method
    )

    if k_fold > 0:
        kf = KFold(n_splits=k_fold)
        kf.get_n_splits(x_train)
        vali_scores = []  # Save the results of each k verification
        num = 0
        for train_index, vali_index in kf.split(x_train):
            train_X, vali_X = x_train[train_index], x_train[vali_index]
            train_y, vali_y = y_train[train_index], y_train[vali_index]
            model = RandomForestClassifier(n_estimators=100,
                                           class_weight='balanced')
            model.fit(train_X, train_y)
            pred = model.predict(vali_X)
            score = precision_score(vali_y, pred, average='macro')  # Record the training effect
            print("Fold " + str(num + 1) + " ============> Precision:" + str(round(score, 4)))
            num += 1
            vali_scores.append(score)
        vali_score = np.average(vali_scores)  # Average value of K-fold verification
        print("Fold_avg ============> Precision:" + str(round(vali_score, 4)))

    if upsample_method == 'smote':
        upsampler = SMOTE(sampling_strategy='auto', random_state=42)
    elif upsample_method == 'smoteenn':
        upsampler = SMOTEENN(sampling_strategy='auto', random_state=42)
    elif upsample_method == 'smotetomek':
        upsampler = SMOTETomek(sampling_strategy='auto', random_state=42)

    # count_category_number(y_train)

    # if upsample_method != 'none':
    #     x_train, y_train = upsampler.fit_resample(x_train, y_train)
    #     # count_category_number(y_train)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                   random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("Training Score:%f" % model.score(x_train, y_train))
    print("Testing Score:%f" % model.score(x_test, y_test))

    if feat_importance_method == 'modelbased':
        feat_importance = model.feature_importances_
        print("feat importance = " + str(feat_importance))
        for c, i in zip(keys, feat_importance):
            print('feature:{},modelbased importance：{}'.format(c, i))

        # 按重要性对特征进行排序
        sorted_idx = feat_importance.argsort()[::-1]

        # 选择排名前十的特征
        top_k_features = sorted_idx[:top_k]

        # 使用排名前十的特征重新训练模型
        x_train_top_features = x_train[:, top_k_features]
        x_test_top_features = x_test[:, top_k_features]

        clf_top_features = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf_top_features.fit(x_train_top_features, y_train)
        y_pred1 = clf_top_features.predict(x_test_top_features)

        test_acc = precision_score(y_test, y_pred1, average='macro')
        recall = recall_score(y_test, y_pred1, average='macro')
        f1 = f1_score(y_test, y_pred1, average='macro')

        print('modelbased top feature test_acc: ---- >{}'.format(test_acc))
        print('modelbased top feature recall: ---- >{}'.format(recall))
        print('modelbased top feature f1_score: ---- >{}'.format(f1))
    else:
        # 计算 Permutation Feature Importance
        result = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42)

        # 输出每个特征的重要性
        for c, i in zip(keys, result.importances_mean):
            print('feature:{},PFI importance：{}'.format(c, i))

        # 按重要性对特征进行排序
        sorted_idx = result.importances_mean.argsort()[::-1]

        # 选择排名前十的特征
        top_k_features = sorted_idx[:top_k]

        # 使用排名前x的特征重新训练模型
        x_train_top_features = x_train[:, top_k_features]
        x_test_top_features = x_test[:, top_k_features]

        clf_top_features = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf_top_features.fit(x_train_top_features, y_train)
        y_pred1 = clf_top_features.predict(x_test_top_features)

        test_acc = precision_score(y_test, y_pred1, average='macro')
        recall = recall_score(y_test, y_pred1, average='macro')
        f1 = f1_score(y_test, y_pred1, average='macro')

        print('PFI top feature test_acc: ---- >{}'.format(test_acc))
        print('PFI top feature recall: ---- >{}'.format(recall))
        print('PFI top feature f1_score: ---- >{}'.format(f1))

    if usehyparams_search:
        model = RandomForestClassifier(class_weight='balanced', random_state=42)

        # 定义随机森林回归器的超参数搜索空间
        param_space = {
            "n_estimators": Integer(20, 300),
            "max_depth": Integer(1, 100),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(1, 10),
            "max_features": Real(0.1, 1.0),
        }

        # 使用贝叶斯优化调整超参数
        opt = BayesSearchCV(
            model,
            param_space,
            n_iter=100

            ,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        opt.fit(np.row_stack((x_train_top_features,x_test_top_features)), np.concatenate((y_train,y_test)))

        search_res = np.column_stack((np.array(opt.optimizer_results_[0]['x_iters']), -1 * opt.optimizer_results_[0]['func_vals']))

        # 转换为 DataFrame
        column_names = [i for i in opt.best_params_.keys()] + ['f1-score']
        search_res_df = pd.DataFrame(search_res, columns=column_names)

        # 将 DataFrame 保存为 CSV 文件
        search_res_df.to_csv(f'{sheet_name}-search_res.csv', index=False)

        model.set_params(**opt.best_params_)
        print('best-f1:',opt.best_score_)
        print('best-params:',opt.best_params_)
        model.fit(x_train_top_features, y_train)

    # (pd.Series(np.array(keys)[sorted_idx][:top_k])).to_csv(
    #     f'../parameter/{sheet_name}-{feat_importance_method}-top{top_k}.csv')

    # if f1_score(y_test, y_pred, average='macro') > f1:
    all_test_acc = precision_score(y_test, y_pred, average='macro')
    all_recall = recall_score(y_test, y_pred, average='macro')
    all_f1 = f1_score(y_test, y_pred, average='macro')
    # else:
    # y_pred = y_pred1

    print('all features test_acc: ---- >{}'.format(all_test_acc))
    print('all features recall: ---- >{}'.format(all_recall))
    print('all features f1_score: ---- >{}'.format(all_f1))

    gbdt_log = r'../result/Random_forest_{}_log.txt'.format(state)
    file_handle = open(gbdt_log, mode='a+')
    file_handle.write('test_acc:{}, recall:{}, f1_score:{}\n'.format(
        test_acc, recall, f1
    ))
    file_handle.close()

    if state == 'magma':
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred,
                              savename=r"../result/Confusion_Matrix_Random_forest_{}.png".format(state),
                              title="Confusion_Matrix_Random_forest_{}".format(state),
                              classes=['AR', 'BR', 'Carbonatite', 'IR', 'Kimberlite', 'Pegmatite'])

    else:
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred,
                              savename=r"../result/Confusion_Matrix_Random_forest_{}.png".format(state),
                              title="Confusion_Matrix_Random_forest_{}".format(state),
                              classes=['Porphyry-type Cu-Au-Mo deposit',
                                       'Skarn-type polymetallic deposit',
                                       'Intrusion-related Au deposit',
                                       'Skarn-type Fe-Cu deposit',
                                       'Nb-Ta deposit'])

    return test_acc, recall, f1, all_test_acc, all_recall, all_f1


if __name__ == '__main__':
    # Define lists of methods to try
    usehyparams_search = True # 贝叶斯调参数
    normal_methods = ['z-score'] # 标准化方法
    impute_methods = ['knn-classification'] # 空缺值填充方法
    upsample_methods = ['smote'] # 数据标签平衡方法
    feat_importance_methods = ['pfi'] # 筛选特征重要性

    top_ks = [8] # 按照重要性选取特征数量\




    for normal_method in normal_methods:
        for impute_method in impute_methods:
            for upsample_method in upsample_methods:
                for feat_importance_method in feat_importance_methods:
                    for top_k in top_ks:
                        with mlflow.start_run():



    # normal_method = 'none'    # 可选 'z-score', 'log', 'minmax'
    # impute_method = 'knn-classification'  # 可选'knn-classification','iterative'
    # upsample_method = 'none' # 可选 'smote'
    #
    # feat_importance_method = 'pfi'
    # top_k = 20  # i for i in range(1, 20)

                            # mlflow.set_experiment('random_forest-Igneous Rocks-pfi')
                            mlflow.set_experiment('random_forest-Ore Deposits-all')
                            mlflow.log_param('normal_method', normal_method)
                            mlflow.log_param('impute_method', impute_method)
                            mlflow.log_param('upsample_method', upsample_method)
                            mlflow.log_param('feat_importance_method', feat_importance_method)
                            mlflow.log_param('top_k', top_k)

                            data_dir = r'../Zircon_Composition_Database.xlsx'
                            # Change sheet name
                            # sheet_name = 'Igneous Rocks Database'
                            sheet_name = 'Ore Deposits Database'
                            k_fold = 5  # Cross validation, if k_Fold<=0, no cross validation
                            miss_ratio = 0.2  # Discard greater than miss_ Ratio missing data
                            n_neighbors = 5  # If n_neighbors=0, select median, else select KNN
                            chose_feat = ['filter', 0]  # 'filter', 'PCA', 'LDA'
                            # chose_feat three methods：
                            # The first is' filter 'to filter out the variance threshold, and the value is the threshold size, which is generally 0.
                            # The second is the 'PCA' dimensionality reduction method, the value is the number of dimensionality reduction features, which can be set freely.
                            # The third is the 'LDA' dimensionality reduction method, the value is the number of dimensionality reduction features, which can be set freely.

                            # Note that if you choose PCA or LDA,the name of the important features will change.
                            # When choose LDA, dimension reduced array dimension cannot be greater than min (n_features, n_classes - 1)
                            # The dimension reduction dimension of deposit cannot be greater than 4, and that of magma cannot be greater than 5

                            print(f'random forest use {sheet_name}, k_fold {k_fold}, miss_ratio {miss_ratio}')

                            test_acc, recall, f1, all_test_acc, all_recall, all_f1 = train(data_file=data_dir,
                                                                  sheet_name=sheet_name,
                                                                  k_fold=k_fold,
                                                                  miss_ratio=miss_ratio,
                                                                  n_neighbors=n_neighbors,
                                                                  chose_featrue=chose_feat,
                                                                  normal_method=normal_method,
                                                                  impute_method=impute_method,
                                                                  upsample_method=upsample_method,
                                                                  usehyparams_search=usehyparams_search,
                                                                  feat_importance_method=feat_importance_method,
                                                                  top_k=top_k)

                            # 记录模型性能和结果
                            mlflow.log_metric('part_accuracy', test_acc)
                            mlflow.log_metric('part_recall', recall)
                            mlflow.log_metric('part_f1', f1)
                            mlflow.log_metric('all_accuracy', all_test_acc)
                            mlflow.log_metric('all_recall', all_recall)
                            mlflow.log_metric('all_f1', all_f1)
                            # mlflow.log_artifact(saved_model)
