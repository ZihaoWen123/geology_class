import pandas as pd
from pylab import *
import math
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



plt.rcParams['axes.unicode_minus'] = False  # Used to display the minus sign normally


def data_process(data_file, sheet_name, miss_ratio, n_neighbors,
                 chose_featrue,
                 is_deep=False,
                 is_infer=False,
                 infer=False):

    def collect_na_value(dataframe):  # count the missing value
        return dataframe.isna().sum() / dataframe.shape[0]

    def tofloat(x): # Remove commas and strings
        try:
            return float(str(x).replace(',', '').replace('**', '00'))
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

    df = pd.read_excel(data_file, sheet_name=sheet_name)
    collect = collect_na_value(pd.DataFrame(df))
    for head, ratio in zip(df.keys(), collect):
        if float(ratio) >= miss_ratio:
            df.drop(head, axis=1, inplace=True)
    df_keys = [key for key in df.keys() if key != 'Rock Type']

    imputer = KNNImputer(n_neighbors=n_neighbors)

    for i, key in enumerate(df):
        if i != 0:
            df[key] = df[key].apply(lambda x: tofloat(x))
            if n_neighbors > 0:
                print(f'pad use knn')
                df[key] = imputer.fit_transform(np.array(df[key]).reshape(-1, 1))
            else:
                print(f'pad use median')
                df.loc[:, str(key)] = df.loc[:, str(key)].fillna(df.loc[:, str(key)].median())
    array_data = np.array(df)
    print('array', array_data.shape)

    if infer is True:
        if sheet_name =='Igneous Rocks Database':
            other = array_data
            target = ['AR'] * len(other)
        else:
            other = array_data
            target = ['Nb-Ta deposit'] * len(other)

    else:
        other, target = array_data[:, 1:], array_data[:, 0].tolist()
# Select feature processing method
    if chose_featrue[0] == 'filter':
        var = VarianceThreshold(threshold=chose_featrue[1])
        other = var.fit_transform(other)

    if chose_featrue[0] == 'PCA':
        other = PCA(n_components=chose_featrue[1]).fit_transform(other)
        df_keys = [i for i in range(len(other))]
    if chose_featrue[0] == 'LDA':

        other = LDA(n_components=chose_featrue[1]).fit_transform(other, target)
        df_keys = [i for i in range(len(other))]

    if sheet_name == 'Igneous Rocks Database':
        other = (other - 0.0) / (130314.9194 - 0.0)
    elif sheet_name == 'Ore Deposits Database':
        other = (other - 0.0) / (77180.76 - 0.0)

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
    size = math.floor(0.9 * len(data_set))  # Split training set and test set
    train_set, test_set = data_set[:size], data_set[size:]
    print(f'{sheet_name} train Set ->:', Counter(train_set[:, -1]))
    print(f'{sheet_name} test Set ->:', Counter(test_set[:, -1]))
    if is_deep is True:
        return train_set, test_set, df_keys, None, None
    else:
        if is_infer is False:
            x_train, x_test, y_train, y_test = \
                train_set[:, :-1], test_set[:, :-1], train_set[:, -1].astype(int), test_set[:, -1].astype(int),
            return x_train, x_test, y_train, y_test, df_keys
        else:
            assert is_infer is True
            return data_set



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



