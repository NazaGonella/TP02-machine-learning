import pandas as pd
import numpy as np
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

cell_diagnosis_dev : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema1/data/raw/cell_diagnosis_dev.csv')
cell_diagnosis_dev_imbalanced : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema1/data/raw/cell_diagnosis_dev_imbalanced.csv')
cell_diagnosis_test : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema1/data/raw/cell_diagnosis_test.csv')
cell_diagnosis_test_imbalanced : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema1/data/raw/cell_diagnosis_test_imbalanced.csv')

def correct_data_types(df : pd.DataFrame) -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    _df = _df.convert_dtypes()
    _df['CellType'] = _df['CellType'].replace('???', 'Unknown')
    _df['GeneticMutation'] = _df['GeneticMutation'].replace('Presnt', '1').replace('Absnt', '').astype(bool)
    _df['Diagnosis'] = _df['Diagnosis'].astype(bool)
    return _df

def remove_na_rows(df : pd.DataFrame) -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    _df = _df.dropna(inplace=False)
    return _df

def fill_na_values(df : pd.DataFrame) -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    numeric_columns : pd.Index = _df.select_dtypes(include='number').columns
    df_median : pd.Series = _df.median(numeric_only=True)

    # Seteo los valores negativos a 0. Luego limito los valores de ciertas features a su rango.
    for col in numeric_columns:
        _df.loc[_df[col] < 0, col] = df_median[col]
    _df.loc[_df['CellAdhesion'] > 1, 'CellAdhesion'] = df_median['CellAdhesion']                            # CellAdhesion          -   Rango (0-1)
    _df.loc[_df['NuclearMembrane'] < 1, 'NuclearMembrane'] = df_median['NuclearMembrane']                   # NuclearMembrane       -   Rango (1-5)
    _df.loc[_df['NuclearMembrane'] > 5, 'NuclearMembrane'] = df_median['NuclearMembrane']   
    _df.loc[_df['Vascularization'] < 0, 'Vascularization'] = df_median['Vascularization']                   # Vascularization       -   Rango (0-10)
    _df.loc[_df['Vascularization'] > 10, 'Vascularization'] = df_median['Vascularization']                
    _df.loc[_df['InflammationMarkers'] < 0, 'InflammationMarkers'] = df_median['InflammationMarkers']       # InflammationMarkers   -   Rango (0-100)
    _df.loc[_df['InflammationMarkers'] > 100, 'InflammationMarkers'] = df_median['InflammationMarkers']   

    _df = _df.fillna(value=df_median)
    _df['CellType'] = _df['CellType'].fillna('Unknown')
    return _df

def standardize_numeric_columns(df : pd.DataFrame) -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    numeric_columns : pd.Index = _df.select_dtypes(include=np.number).columns
    _df[numeric_columns] = (_df[numeric_columns] - _df[numeric_columns].mean()) / _df[numeric_columns].std()
    return _df

def one_hot_encoding(df : pd.DataFrame, column : str) -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    return pd.get_dummies(_df, prefix=[column], dtype=bool)

def process_and_stardardize(df : pd.DataFrame, filename : str = "", save_path : str = "") -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    df_processed : pd.DataFrame = correct_data_types(_df)
    df_processed = fill_na_values(df_processed)
    df_processed = one_hot_encoding(df_processed, 'CellType')
    df_processed_and_standardized : pd.DataFrame = standardize_numeric_columns(df_processed)
    if save_path and filename:
        df_processed.to_csv(f'{save_path}/{filename}_processed.csv', index=False)
        df_processed_and_standardized.to_csv(f'{save_path}/{filename}_processed_and_standardized.csv')
    return df_processed_and_standardized

def undersample(df: pd.DataFrame, objective_class: str = '') -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    class_0 : pd.DataFrame = _df[df[objective_class] == 0]
    class_1 : pd.DataFrame = _df[df[objective_class] == 1]
    majority : pd.DataFrame = class_0 if len(class_0) > len(class_1) else class_1
    minority : pd.DataFrame = class_1 if len(class_0) > len(class_1) else class_0
    majority_sampled : pd.DataFrame = majority.sample(len(minority), random_state=42)
    return pd.concat([minority, majority_sampled]).sample(frac=1) # shuffleo

def oversample_by_duplication(df: pd.DataFrame, objective_class: str = '') -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    class_0 : pd.DataFrame = _df[df[objective_class] == 0]
    class_1 : pd.DataFrame = _df[df[objective_class] == 1]
    majority : pd.DataFrame = class_0 if len(class_0) > len(class_1) else class_1
    minority : pd.DataFrame = class_1 if len(class_0) > len(class_1) else class_0
    minority_sampled : pd.DataFrame = minority.sample(len(majority) - len(minority), replace=True, random_state=42) # replace=True permite samplear una fila mas de una vez
    # minority.info(verbose=False)
    # print(minority_sampled)
    return pd.concat([minority, minority_sampled, majority]).sample(frac=1) # shuffleo

# def oversample_by_SMOTE(df: pd.DataFrame, objective_class: str = '', k : int = 2) -> pd.DataFrame:
#     _df : pd.DataFrame = df.copy()
#     class_0 : pd.DataFrame = _df[df[objective_class] == 0]
#     class_1 : pd.DataFrame = _df[df[objective_class] == 1]
#     majority : pd.DataFrame = class_0 if len(class_0) > len(class_1) else class_1
#     minority : pd.DataFrame = class_1 if len(class_0) > len(class_1) else class_0
#     numeric_columns : pd.Index = df.select_dtypes(include=np.number).columns
#     boolean_columns : pd.Index = df.select_dtypes(include=bool).columns
#     new_samples : list[pd.Series] = []
#     for i in range(len(majority) - len(minority)):
#         random_sample : pd.Series = minority.sample(1, random_state=42+i).iloc[0]   # Me quedo con la primer y única fila del DataFrame que devuelve sample()
#         for n in numeric_columns.values:
#             n_column_matrix : np.ndarray = _df[n].to_numpy()
#             random_sample_vector : np.ndarray = random_sample.to_numpy()
#             distance_column : np.ndarray = np.linalg.norm(n_column_matrix - random_sample_vector)
#             k_indices : np.ndarray = np.argpartition(distance_column, k+1, axis=1)[:, :k+1]
#             for j in k_indices:
#                 nearest_sample : pd.Series = minority.iloc[j]
#                 new_sample : pd.Series = 

#     return pd.DataFrame()

# def oversampling_SMOTE(df : pd.DataFrame, col ,seed,k=5):
#     np.random.seed(seed)
#     df0=df[df[col]==0].copy()
#     df1=df[df[col]==1].copy()
#     if df0.shape[0]>df1.shape[0]:
#         df0,df1=df1,df0
    
#     while df0.shape[0]<df1.shape[0]:
#         i=np.random.randint(0,df0.shape[0])
#         nodechosen=df0.iloc[i]
#         distances = np.linalg.norm(df0.values - nodechosen.values, axis=1)
#         knn=np.argsort(distances)[1:k+1]
#         j= np.random.randint(0,k)
#         nnchosen=df0.iloc[knn[j]]
#         newrow=nodechosen + (nnchosen-nodechosen) * np.random.rand()
#         newrow = pd.DataFrame([newrow])
#         df0 = pd.concat([df0, newrow], axis=0)
        
#     return pd.concat([df0,df1],axis=0)

def oversample_by_SMOTE(df: pd.DataFrame, objective_class: str = '', k : int = 2):
    _df : pd.DataFrame = df.copy()
    class_0 : pd.DataFrame = _df[df[objective_class] == 0]
    class_1 : pd.DataFrame = _df[df[objective_class] == 1]
    majority : pd.DataFrame = class_0 if len(class_0) > len(class_1) else class_1
    minority : pd.DataFrame = class_1 if len(class_0) > len(class_1) else class_0
    numeric_columns : pd.Index = df.select_dtypes(include=np.number).columns
    boolean_columns : pd.Index = df.select_dtypes(include=bool).columns
    new_samples : pd.DataFrame = minority.copy()
    for i in range(len(majority) - len(minority)):
        random_minority_sample : pd.DataFrame = minority.sample(n=1, random_state=42+i)
        distances : np.ndarray[float] = np.linalg.norm(np.array(minority[numeric_columns].values, dtype=np.float64) - np.array(random_minority_sample[numeric_columns].values, dtype=np.float64), axis=1)
        # print(len(distances))
        knn_indeces : np.ndarray[int] = np.argsort(distances)[1:k+1]
        nearest_neighbor : pd.DataFrame = minority.iloc[knn_indeces[np.random.randint(0,k)]]
        # new_sample : np.ndarray = (np.array(random_minority_sample[numeric_columns].iloc[0], dtype=np.float64) + np.array(nearest_neighbor[numeric_columns].iloc[0], dtype=np.float64)) / 2
        # new_sample = pd.DataFrame([new_sample], columns=random_minority_sample[numeric_columns].columns)
        # new_sample[boolean_columns] = random_minority_sample[boolean_columns].astype(bool)
        new_numeric_values = (
            np.array(random_minority_sample[numeric_columns].iloc[0], dtype=np.float64) +
            np.array(nearest_neighbor[numeric_columns].iloc[0], dtype=np.float64)
        ) / 2
        new_sample = pd.DataFrame([new_numeric_values], columns=numeric_columns)
        for col in boolean_columns:
            new_sample[col] = bool(random_minority_sample[col].iloc[0])
        # print(new_sample)
        new_samples = pd.concat([new_samples, new_sample], axis=0)
    minority = pd.concat([minority, new_samples], axis=0)
    return pd.concat([minority, majority],axis=0)

# Realizo el rebalanceo con el dataset preprocesado
cell_diagnosis_dev_imbalanced_processed_and_standardized : pd.DataFrame = process_and_stardardize(
    cell_diagnosis_dev_imbalanced, 
    filename='cell_diagnosis_dev_imbalanced', 
    save_path=f'{project_root}/TP02/problema1/data/processed/'
)
# print("---------------------------------------------------------------")
# print("OVERSAMPLING MEDIANTE SMOTE")
# print("---------------------------------------------------------------")
# print(cell_diagnosis_dev_imbalanced_processed_and_standardized.info())
cell_diagnosis_dev_oversampled_by_SMOTE : pd.DataFrame = oversample_by_SMOTE(cell_diagnosis_dev_imbalanced_processed_and_standardized, objective_class='Diagnosis', k=2)
# print(cell_diagnosis_dev_oversampled_by_SMOTE.info())
for col in ['Diagnosis', 'GeneticMutation', 'CellType_Epthlial', 'CellType_Mesnchymal', 'CellType_Unknown']:
    print(f"{col}: {cell_diagnosis_dev_oversampled_by_SMOTE[col].unique()}")
print(cell_diagnosis_dev_oversampled_by_SMOTE.info())