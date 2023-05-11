import numpy as np
import pandas as pd
from pylab import *
import math
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

plt.rcParams['axes.unicode_minus'] = False  # Used to display the minus sign normally


def data_process(data_file, sheet_name, miss_ratio, n_neighbors,
                 chose_featrue,
                 normal_method,
                 impute_method,
                 is_deep=False,
                 is_infer=True,
                 ):
    def collect_na_value(dataframe):  # count the missing value
        return dataframe.isna().sum() / dataframe.shape[0]

    def tofloat(x):
        try:
            return float(str(x).replace(',', '').replace('**', '00'))  # Remove commas and strings
        except:
            return None

    def label2idx(label_data):
        if sheet_name == 'Igneous Rocks Database':
            label = ['AR', 'BR', 'Carbonatite', 'IR', 'Kimberlite', 'Pegmatite']
        else:
            label = ['Porphyry-type Cu-Au-Mo deposit', 'Skarn-type polymetallic deposit',
                     'Intrusion-related Au deposit', 'Skarn-type Fe-Cu deposit', 'Nb-Ta deposit']
        new_label = []
        for x in label_data:
            new_label.append(label.index(x))
        new_label = np.array(new_label).reshape(len(label_data), 1)
        return new_label

    apply_type = sheet_name
    tmp = pd.read_excel(data_file, sheet_name=sheet_name)
    if apply_type == 'Igneous Rocks Database':
        data_file1  = './Zircon_Composition_Database.xlsx'
        sheet_name1 = 'Igneous Rocks Database'
        used_features = list(tmp.columns)
    else:
        data_file1 = './Zircon_Composition_Database.xlsx'
        sheet_name1 = 'Ore Deposits Database'
        used_features = list(tmp.columns)

    df = pd.read_excel(data_file1, sheet_name=sheet_name1)
    test_df = df.iloc[:tmp.shape[0], :].copy()
    test_df[used_features] = tmp.values

    df.drop(['La(ppm)', 'Pr(ppm)'], axis=1, inplace=True)
    test_df.drop(['La(ppm)', 'Pr(ppm)', 'Rock Type'], axis=1, inplace=True)

    collect = collect_na_value(pd.DataFrame(df))
    for head, ratio in zip(df.keys(), collect):
        if float(ratio) >= miss_ratio:
            df.drop(head, axis=1, inplace=True)
            test_df.drop(head, axis=1, inplace=True)

    df_keys = [key for key in df.keys() if key != 'Rock Type']

    for i, key in enumerate(df):
        if i != 0:
            df[key] = df[key].apply(lambda x: tofloat(x))

    test_df_keys = [key for key in test_df.keys()]
    used_features_idx = [test_df_keys.index(f) for f in used_features]

    for i, key in enumerate(test_df):
        test_df[key] = test_df[key].apply(lambda x: tofloat(x))

    test_df.fillna(df[1:].median(), inplace=True)

    if impute_method == 'knn':
        # imputer = KNNImputer(n_neighbors=n_neighbors)
        # df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:].values.astype(np.float))
        imputer = KNNImputer(n_neighbors=n_neighbors)
        for i, key in enumerate(df):
            if i != 0:
                df[key] = df[key].apply(lambda x: tofloat(x))
                if n_neighbors > 0:
                    print(f'pad use knn')
                    df[key] = imputer.fit_transform(np.array(df[key]).reshape(-1, 1))
                    # test_df[key] = imputer.transform(np.array(test_df[key]).reshape(-1, 1))
                else:
                    print(f'pad use median')
                    df.loc[:, str(key)] = df.loc[:, str(key)].fillna(df.loc[:, str(key)].median())
                    # test_df.loc[:, str(key)] = test_df.loc[:, str(key)].fillna(test_df.loc[:, str(key)].median())
    elif impute_method == 'iterative':
        # 将数据缩放到(0,1)范围内
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df.iloc[:, 1:].values.astype(np.float))

        # 使用Random Forest回归器作为基本估计器进行迭代插补
        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=42)
        imputed_data = imputer.fit_transform(scaled_data)

        # 将数据缩放回原始范围
        imputed_data = scaler.inverse_transform(imputed_data)

        # 将结果转换为Pandas DataFrame
        df.iloc[:, 1:] = imputed_data
    elif impute_method == 'knn-classification':
        # 根据标签对数据按类别分组
        groups = df.groupby('Rock Type')

        # 初始化 KNNImputer
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # 对每个组应用 KNN 插补，并将结果存储在一个新的 DataFrame 中
        imputed_dfs = []
        for label, group in groups:
            for i, key in enumerate(group):
                if i != 0:
                    if n_neighbors > 0:
                        print(f'pad use knn-classification')
                        group[key] = imputer.fit_transform(np.array(group[key]).reshape(-1, 1))
                    else:
                        print(f'pad use median')
                        group.loc[:, str(key)] = group.loc[:, str(key)].fillna(group.loc[:, str(key)].median())

            imputed_dfs.append(group)

        # 将插补后的数据组合到一个新的 DataFrame中
        df = pd.concat(imputed_dfs, ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
    elif impute_method == 'none':
        # 将空缺值替换为np.nan
        df.fillna(np.nan)
        # df.replace(to_replace=None, value=np.nan, inplace=True)

    array_data = np.array(df)
    np.random.seed(2022)
    np.random.shuffle(array_data)
    other, target = array_data[:, 1:], array_data[:, 0].tolist()

    if chose_featrue[0] == 'filter':
        var = VarianceThreshold(threshold=chose_featrue[1])
        other = var.fit_transform(other)
        test_dataset = var.transform(test_df.values)


    if chose_featrue[0] == 'PCA':
        other = PCA(n_components=chose_featrue[1]).fit_transform(other)
        df_keys = [i for i in range(len(other))]
    if chose_featrue[0] == 'LDA':
        other = LDA(n_components=chose_featrue[1]).fit_transform(other, target)
        df_keys = [i for i in range(len(other))]

    if normal_method == 'minmax':
        scaler = MinMaxScaler()
    elif normal_method == 'z-score':
        scaler = StandardScaler()
    elif normal_method == 'log':
        scaler = PowerTransformer()
    else:
        scaler = None

    if normal_method != 'none':
        other = scaler.fit_transform(other)
        test_dataset = scaler.transform(test_dataset)

    if sheet_name == 'Ore Deposits Database':
        deposits = []
        for deposit_target in target:
            if '/' in deposit_target:
                L = deposit_target.split('/')[0]
                deposits.append(L.strip())
            else:
                deposits.append(deposit_target.strip())
        target_ = deposits
    else:
        target_ = target
    target = label2idx(target_)
    data_set = np.hstack((other, target)).astype(float)

    size = math.floor(0.9 * len(data_set))
    train_set, test_set = data_set[:size], data_set[size:]
    # print(f'{sheet_name} train Set ->:', Counter(train_set[:, -1]))
    # print(f'{sheet_name} test Set ->:', Counter(test_set[:, -1]))
    if is_deep is True:
        return train_set, test_set, df_keys, None, None
    else:
        if is_infer is False:
            x_train, x_test, y_train, y_test = \
                train_set[:, :-1], test_set[:, :-1], train_set[:, -1].astype(int), test_set[:, -1].astype(int),
            return x_train, x_test, y_train, y_test, df_keys
        else:
            assert is_infer is True
            return test_dataset[:, used_features_idx]


class BatchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]
        train_set = data[:-1]
        train_set = torch.tensor(
            train_set, dtype=torch.float32
        ).unsqueeze(0)
        train_set = train_set.repeat(1, train_set.shape[-1], 1)
        label_set = torch.tensor(int(data[-1]))
        return train_set, label_set

    def __len__(self):
        return len(self.data)



