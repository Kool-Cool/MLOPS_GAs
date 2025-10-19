# test_evaluation.py
import pytest
import joblib
import numpy as np

def test_model_prediction():
    model = joblib.load('model/model.joblib')
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred = model.predict(sample)
    assert pred is not None
    assert len(pred) == 1
