# test_validation.py
import pytest
import pandas as pd

def test_data_shape():
    df = pd.read_csv("data/iris.csv")
    assert df.shape[1] == 5  # 4 features + 1 label
    assert df.isnull().sum().sum() == 0
