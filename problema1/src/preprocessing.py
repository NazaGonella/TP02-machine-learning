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