from data_process import data_process
import joblib
import pandas as pd
from collections import Counter


def load_data(data_file, sheet_name, miss_ratio, n_neighbors, chose_featrue, infer):
    dataset = data_process(
        data_file=data_file,
        sheet_name=sheet_name,
        miss_ratio=miss_ratio,
        n_neighbors=n_neighbors,
        chose_featrue=chose_featrue,
        infer=infer,
        is_deep=False,
        is_infer=True,

    )
    return dataset


def load_model(parameter_path):
    model = joblib.load(parameter_path)
    return model


def main(parameter_path, data_file, sheet_name, miss_ratio, n_neighbors, chose_featrue):
    model = load_model(parameter_path)
    data = load_data(data_file, sheet_name, miss_ratio, n_neighbors, chose_featrue, infer=True)
    x_test, y_test = data[:, :-1], data[:, -1].astype(int)
    y_pred = model.predict(x_test)

    # Save forecast result table: including element column and forecast result column
    # You need to change case data path here
    # td = pd.read_excel(r'Yilgarn_zircon_chem/2.xlsx')
    td = pd.read_excel(r'Deposit_case_test/kuangchuang  test.xlsx')
    if sheet_name=='Igneous Rocks Database':
        result = pd.DataFrame(y_pred)
        tr = pd.concat([td, result], axis=1)
        tr.rename(columns={0: 'result'}, inplace=True)
        tr.replace({"result": {0: 'AR', 1: 'BR',2: 'Carbonatite',3: 'IR',4: 'Kimberlite',5: 'Pegmatite'}}, inplace=True)
        tr.to_excel("Igneous_result/case_analysis1.xlsx", index=False)
        print(tr)
    else:
        result = pd.DataFrame(y_pred)
        tr = pd.concat([td, result], axis=1)
        tr.rename(columns={0 : 'result'}, inplace=True)
        tr.replace({"result" : {0 : 'Porphyry-type Cu-Au-Mo deposit',
                                1 : 'Skarn-type polymetallic deposit',
                                2 : 'Intrusion-related Au deposit',
                                3 : 'Skarn-type Fe-Cu deposit',
                                4 : 'Nb-Ta deposit'}},
                   inplace=True)
        tr.to_excel("Deposit_result/test_analysis1.xlsx", index=False)
        print(tr)

    # Calculate the percentage of results
    if sheet_name == 'Igneous Rocks Database' :
        result = Counter(y_pred)
        result_count = pd.DataFrame(list(result.items()))
        result_count.rename(columns={0: 'Rock_Type', 1: 'count'}, inplace=True)
        total = []
        for i in range(len(set(y_pred))):
            total.append(len(y_pred))
        result_count['total'] = pd.DataFrame(total)
        result_count.replace({"Rock_Type": {0: 'AR',
                                            1: 'BR',
                                            2: 'Carbonatite',
                                            3: 'IR',
                                            4: 'Kimberlite',
                                            5: 'Pegmatite'}},
                             inplace=True)
        result_count['percent'] = result_count['count'] / result_count['total']
        result_count.drop('total', axis=1, inplace=True)
        result_count['percent'] = result_count['percent'].apply(lambda x: format(x, '.1%'))
        print(result_count)
    else:
        result = Counter(y_pred)
        result_count = pd.DataFrame(list(result.items()))
        result_count.rename(columns={0 : 'Deposit_Type', 1 : 'count'}, inplace=True)
        total = []
        for i in range(len(set(y_pred))) :
            total.append(len(y_pred))
        result_count['total'] = pd.DataFrame(total)
        result_count.replace(
            {"Deposit_Type" : {0 : 'Porphyry-type Cu-Au-Mo deposit',
                            1 : 'Skarn-type polymetallic deposit',
                            2 : 'Intrusion-related Au deposit',
                            3 : 'Skarn-type Fe-Cu deposit',
                            4 : 'Nb-Ta deposit'}},
            inplace=True)
        result_count['percent'] = result_count['count'] / result_count['total']
        result_count.drop('total', axis=1, inplace=True)
        result_count['percent'] = result_count['percent'].apply(lambda x : format(x, '.1%'))
        print(result_count)


if __name__ == '__main__':
    # Choose trained model

    # parameter_path_ = r'parameter/RandomForest-Igneous Rocks Database-5-0.8-5.pkl'
    # data_file_ = r'Yilgarn_zircon_chem/2.xlsx'# You need to change case data path here
    # sheet_name_ = 'Igneous Rocks Database' # Change the sheet name

    parameter_path_ = r'parameter/RandomForest-Ore Deposits Database-5-0.8-5.pkl'
    data_file_ = r'Deposit_case_test/kuangchuang  test.xlsx' # You need to change case data path here
    sheet_name_ = 'Ore Deposits Database' # Change the sheet name

    miss_ratio_ = 1
    n_neighbors_ = 5
    chose_featrue_ = ['filter', 0]  # 'filter', 'PCA', 'LDA'

    main(parameter_path=parameter_path_,
         data_file=data_file_,
         sheet_name=sheet_name_,
         miss_ratio=miss_ratio_,
         n_neighbors=n_neighbors_,
         chose_featrue=chose_featrue_)