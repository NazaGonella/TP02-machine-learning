import pandas as pd
import numpy as np

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
    _df = _df.fillna(value=_df.median(numeric_only=True))
    _df['CellType'] = _df['CellType'].fillna('Unknown')
    return _df

def standardize_numeric_columns(df : pd.DataFrame) -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    numeric_columns : pd.DataFrame = _df.select_dtypes(include=np.number).columns
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
    class_0 : pd.Series = df[df[objective_class] == 0]
    class_1 : pd.Series = df[df[objective_class] == 1]
    majority : pd.Series = class_0 if len(class_0) > len(class_1) else class_1
    minority : pd.Series = class_1 if len(class_0) > len(class_1) else class_0
    majority_sampled : pd.Series = majority.sample(len(minority), random_state=42)
    return pd.concat([minority, majority_sampled]).sample(frac=1) # shuffleo

def oversample_by_duplication(df: pd.DataFrame, objective_class: str = '') -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    class_0 : pd.Series = df[df[objective_class] == 0]
    class_1 : pd.Series = df[df[objective_class] == 1]
    majority : pd.Series = class_0 if len(class_0) > len(class_1) else class_1
    minority : pd.Series = class_1 if len(class_0) > len(class_1) else class_0
    minority_sampled : pd.Series = minority.sample(len(majority) - len(minority), replace=True, random_state=42) # replace=True permite samplear una fila mas de una vez
    return pd.concat([minority, minority_sampled, majority]).sample(frac=1) # shuffleo