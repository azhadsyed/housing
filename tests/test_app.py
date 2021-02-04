import pickle

import pytest
from web import app
from unittest.mock import patch

geopy_mock_response = pickle.load(open("data/geopy_mock_response.pkl", "rb"))


@pytest.fixture
def form_data():
    return {
        "address": "397 Bridge St., Brooklyn NY",
        "cats_ok": False,
        "dogs_ok": False,
        "housing_type": "apartment",
        "laundry": "laundry in bldg",
        "bedrooms": 2,
        "bathrooms": 1,
        "parking": "attached garage",
        "no_smoking": False,
        "is_furnished": True,
        "wheelchair_acccess": False,
        "ev_charging": False,
    }


@patch("web.app.geocode_address", return_value=geopy_mock_response)
def test_form_data_to_dataframe(mock, form_data):
    # check that test case's # columns matches model requirement
    assert len(form_data.keys()) + 1 == len(app.model_columns)

    dataframe = app.form_data_to_dataframe(form_data)
    assert str(type(dataframe)) == "<class 'pandas.core.frame.DataFrame'>"
    assert "latitude" in dataframe.columns and "longitude" in dataframe.columns
    mock.assert_called_once()


@pytest.fixture
@patch("web.app.geocode_address", return_value=geopy_mock_response)
def dataframe(mock, form_data):
    dataframe = app.form_data_to_dataframe(form_data)
    mock.assert_called_once()
    return dataframe


def test_predict_and_unpack(dataframe):
    estimate, bias, contributions = app.predict_and_unpack(app.model, dataframe)
    assert str(type(estimate)) == "<class 'numpy.float64'>"
    assert str(type(bias)) == "<class 'numpy.float64'>"
    assert type(contributions) == list
    assert abs(estimate - bias - sum([i[1] for i in contributions])) <= 0.01


@pytest.fixture
def prediction(dataframe):
    return app.predict_and_unpack(app.model, dataframe)


def test_clean_features(prediction):
    estimate, bias, contributions = prediction
    cleaned_features = app.clean_features(contributions, app.categorical_features)
    assert type(cleaned_features) == dict
    assert abs(estimate - bias - sum(cleaned_features.values())) <= 0.01


@pytest.fixture
def cleaned_features(prediction):
    contributions = prediction[2]
    return app.clean_features(contributions, app.categorical_features)


def test_order_features(prediction, cleaned_features, form_data):
    bias = prediction[1]
    ordered_features = app.order_features(bias, cleaned_features, form_data)
    assert abs(
        sum([i[1] for i in ordered_features]) - bias - sum(cleaned_features.values())
        <= 0.01
    )


def test_clean_and_style(prediction, form_data):
    bias, contributions = prediction[1], prediction[2]
    explanation = app.clean_and_style(bias, contributions, form_data)
    assert type(explanation) == list
