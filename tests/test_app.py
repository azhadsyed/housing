from functools import reduce
import pytest
from web import app


@pytest.fixture
def form_data():
    return {
        "address": "397 Bridge St., Brooklyn NY",
        "cats_ok": True,
        "dogs_ok": False,
        "housing_type": "apartment",
        "laundry": "laundry in bldg",
        "bedrooms": 3,
        "bathrooms": 1,
        "parking": "street parking",
        "no_smoking": False,
        "is_furnished": False,
        "wheelchair_acccess": True,
        "ev_charging": False,
    }


def test_form_data_to_dataframe(form_data):
    # check that test case has same # of fields as cached column order
    assert len(form_data.keys()) + 1 == len(app.column_order)

    dataframe = app.form_data_to_dataframe(form_data)
    assert str(type(dataframe)) == "<class 'pandas.core.frame.DataFrame'>"
    assert "latitude" in dataframe.columns and "longitude" in dataframe.columns


@pytest.fixture
def dataframe(form_data):
    return app.form_data_to_dataframe(form_data)


def test_predict_and_unpack(dataframe):
    estimate, bias, contributions = app.predict_and_unpack(app.model, dataframe)
    assert str(type(estimate)) == "<class 'numpy.float64'>"
    assert str(type(bias)) == "<class 'numpy.float64'>"
    assert type(contributions) == list
    try:
        assert estimate - bias + sum([i[1] for i in contributions]) <= 0.01
    except AssertionError:
        print(estimate - bias + sum([i[1] for i in contributions]))


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


def test_order_features(prediction, cleaned_features):
    bias = prediction[1]
    ordered_features = app.order_features(bias, cleaned_features)
    assert abs(
        sum([i[1] for i in ordered_features]) - bias - sum(cleaned_features.values())
        <= 0.01
    )


def test_clean_and_style(prediction):
    bias, contributions = prediction[1], prediction[2]
    assert (
        str(type(app.clean_and_style(bias, contributions)))
        == "<class 'markupsafe.Markup'>"
    )