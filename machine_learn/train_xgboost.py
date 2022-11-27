from utils import *
from data_process import data_process
from sklearn.metrics import recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import joblib


def train(data_file, sheet_name, k_fold, miss_ratio, n_neighbors, chose_featrue):
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
        is_deep=False
    )

    if k_fold > 0:
        kf = KFold(n_splits=k_fold)
        kf.get_n_splits(x_train)
        vali_scores = []  # Save the results of each k verification
        num = 0
        for train_index, vali_index in kf.split(x_train):
            train_X, vali_X = x_train[train_index], x_train[vali_index]
            train_y, vali_y = y_train[train_index], y_train[vali_index]
            model = XGBClassifier(learning_rate=0.01,
                                  n_estimators=100,  # Number of trees--10 trees to build xgboost
                                  max_depth=10,  # The depth of the tree
                                  min_child_weight=1,  # Minimum weight of leaf node
                                  gamma=0.,  # Parameters before the number of leaf nodes in the penalty term
                                  subsample=1,  # Establish decision tree for all samples
                                  colsample_btree=1,  # Establish decision tree for all features
                                  scale_pos_weight=1,  # Solve the problem of unbalanced number of samples
                                  random_state=60,  # random number
                                  slient=0,
                                  eval_metric=['logloss', 'auc', 'error']
                                  )
            model.fit(train_X, train_y)
            pred = model.predict(vali_X)
            score = precision_score(vali_y, pred, average='macro')  # Record the training effect
            print("Fold " + str(num + 1) + " ============> Precision:" + str(round(score, 4)))
            num += 1
            vali_scores.append(score)
        vali_score = np.average(vali_scores)  # Average value of K-fold verification
        print("Fold_avg ============> Precision:" + str(round(vali_score, 4)))

    model = XGBClassifier(learning_rate=0.01,
                          n_estimators=100,  # Number of trees--10 trees to build xgboost
                          max_depth=10,  # The depth of the tree
                          min_child_weight=1,  # Minimum weight of leaf node
                          gamma=0.,  # Parameters before the number of leaf nodes in the penalty term
                          subsample=1,  # Establish decision tree for all samples
                          colsample_btree=1,  # Establish decision tree for all features
                          scale_pos_weight=1,  # Solve the problem of unbalanced number of samples
                          random_state=60,  # random number
                          slient=0,
                          eval_metric=['logloss', 'auc', 'error']
                          )

    model.fit(x_train, y_train)
    joblib.dump(model, r'../parameter/xgboost-{}-{}-{}-{}.pkl'.format(
        sheet_name, k_fold, miss_ratio, n_neighbors))

    y_pred = model.predict(x_test)
    feat_importance = model.feature_importances_
    print("feat importance = " + str(feat_importance))

    for c, i in zip(keys, feat_importance):
        print('feature:{},importance：{}'.format(c, i))

    print("Traing Score:%f" % model.score(x_train, y_train))
    print("Testing Score:%f" % model.score(x_test, y_test))

    test_acc = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print('test_acc: ---- >{}'.format(test_acc))
    print('recall: ---- >{}'.format(recall))
    print('f1_score: ---- >{}'.format(f1))

    gbdt_log = r'../result/xgboost_{}_log.txt'.format(state)
    file_handle = open(gbdt_log, mode='a+')
    file_handle.write('test_acc:{}, recall:{}, f1_score:{}\n'.format(
        test_acc, recall, f1
    ))
    file_handle.close()

    if state == 'magma':
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred,
                              savename=r"../result/Confusion_Matrix_xgboost_{}.png".format(state),
                              title="Confusion_Matrix_xgboost_{}".format(state),
                              classes=['AR', 'BR', 'Carbonatite', 'IR', 'Kimberlite', 'Pegmatite'])
    else:
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred,
                              savename=r"../result/Confusion_Matrix_xgboost_{}.png".format(state),
                              title="Confusion_Matrix_xgboost_{}".format(state),
                              classes=['Porphyry-type Cu-Au-Mo deposit',
                                       'Skarn-type polymetallic deposit',
                                       'Intrusion-related Au deposit',
                                       'Skarn-type Fe-Cu deposit',
                                       'Nb-Ta deposit'])


if __name__ == '__main__':
    data_dir = r'../Zircon_Composition_Database.xlsx'

    # sheet_name = 'Igneous Rocks Database'
    sheet_name = 'Ore Deposits Database'
    k_fold = 5  # Cross validation, if k_ Fold<=0, no cross validation
    miss_ratio = 0.8  # Discard greater than miss_ Ratio missing data
    n_neighbors = 5  # If n_neighbors=0, select median, else select KNN
    chose_feat = ['filter', 0]  # 'filter', 'PCA', 'LDA'
    # chose_feat three methods：
    # The first is' filter 'to filter out the variance threshold, and the value is the threshold size, which is generally 0.
    # The second is the 'PCA' dimensionality reduction method, the value is the number of dimensionality reduction features, which can be set freely.
    # The third is the 'LDA' dimensionality reduction method, the value is the number of dimensionality reduction features, which can be set freely.

    # Note that if you choose PCA or LDA,the name of the important features will change.
    # When choose LDA, dimension reduced array dimension cannot be greater than min (n_features, n_classes - 1)
    # The dimension reduction dimension of deposit cannot be greater than 4, and that of magma cannot be greater than 5
    print(f'xgboost use {sheet_name}, k_fold {k_fold}, miss_ratio {miss_ratio}')

    train(data_file=data_dir,
          sheet_name=sheet_name,
          k_fold=k_fold,
          miss_ratio=miss_ratio,
          n_neighbors=n_neighbors,
          chose_featrue=chose_feat)
