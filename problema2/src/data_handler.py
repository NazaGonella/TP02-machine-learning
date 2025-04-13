import pandas as pd
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

def get_train_and_validation_sets(df : pd.DataFrame, train_fraction : float = 0.8, seed : int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train : pd.DataFrame = df.sample(frac=train_fraction,random_state=seed)
    validation : pd.DataFrame = df.drop(train.index)
    return train, validation
