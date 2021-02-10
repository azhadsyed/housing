import pytest

from ml.train import RandomForestModel


def test_rfm_train():
    rfm = RandomForestModel()
    rfm.train_random_forest("data/data.csv", "price", ["id"])