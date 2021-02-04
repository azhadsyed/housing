# warm for the winter - Middle School, Aso

import json
import pickle
from functools import lru_cache as cache

import pandas as pd
from flask import Flask, Markup, render_template, request
from geopy.geocoders import Nominatim
from joblib import load
from treeinterpreter import treeinterpreter as ti
import shap

from .config import options
from .forms import EstimateForm

# from here until first function definition, this could all live in some sort of config?

model_columns = options["column order"]
categorical_features = options["categorical features"]

app = Flask(__name__)
app.config.from_pyfile("config.py")

model = load("data/model.joblib")
feature_names = model.steps[0][1].get_feature_names()
explainer = shap.TreeExplainer(model.steps[1][1])

nominatim = Nominatim(user_agent="nyc_rent_estimator")
all_hands_on_deck = pickle.load(open("data/geopy_mock_response.pkl", "rb"))

ti_enabled = False


@cache(maxsize=200)
def geocode_address(address):
    """Accepts an address string, returns a tuple of latitude and longitude"""
    try:
        response = nominatim.geocode(address)
        print("API UP")
        return response
    except:
        return all_hands_on_deck


def form_data_to_dataframe(form_data):
    if "submit" in form_data.keys():
        del form_data["submit"]

    location = geocode_address(form_data["address"])

    form_data["latitude"] = location.latitude
    form_data["longitude"] = location.longitude
    dict_for_pandas = {k: [v] for k, v in form_data.items()}
    dataframe = pd.DataFrame.from_dict(dict_for_pandas)
    return dataframe[model_columns]


def predict_and_unpack(model, dataframe) -> (float, float, list):
    """Accepts an ML model and a dataframe with one row. Returns a tuple with
    one floating precision estimate, one floating precision bias, and one two-
    dimensional array of feature names and floating precision contributions."""
    if len(dataframe) != 1:
        raise ValueError("dataframe cannot have more than one row.")
    if ti_enabled:
        estimate, bias, contributions = ti.predict(
            model.steps[-1][1], model.steps[0][1].transform(dataframe)
        )
        estimate = estimate[0, 0]
        bias, contributions = bias[0], contributions[0]
        contributions = list(
            filter(lambda x: x[1] != 0, zip(feature_names, contributions))
        )
    else:
        estimate = model.predict(dataframe)[0]
        contributions = explainer.shap_values(model.steps[0][1].transform(dataframe))[0]
        bias = explainer.expected_value[0]
        contributions = list(zip(feature_names, contributions))
    return estimate, bias, contributions


def clean_features(array, order) -> dict:
    """Accepts array of raw feature contributions. Returns a clean dict of
    contributions where variables are summed by category"""
    sums = {i: 0 for i in order}

    for tup in array:
        feature, contribution = tup
        onehot = feature.find("onehotencoder")  # returns -1 if substr not in str
        if onehot != -1:
            index = int(feature[16])
            sums[order[index]] += contribution
        else:
            sums[feature] = contribution
    return sums


def display_pet_status(cats, dogs):
    """Accepts two boolean values for whether cats and dogs are allowed, and
    returns the relevant description of pet status as a string."""
    if not cats and not dogs:
        return "No Pets Allowed"
    elif cats and not dogs:
        return "Cats Allowed"
    elif not cats and dogs:
        return "Dogs Allowed"
    else:
        return "Pets Allowed"


def order_features(bias, features, form_data) -> list:
    """Accepts a clean dictionary of features. Returns a UI-friendly array of
    tuples (ordered) explaining feature contributions."""
    ordered_features = []

    # first establish the base
    ordered_features.append(("Base", bias))

    # then factor in the location
    ordered_features.append(("Location", features["latitude"] + features["longitude"]))

    # then the number of bedrooms/bathrooms
    bedrooms, bathrooms = (form_data["bedrooms"], form_data["bathrooms"])

    ordered_features.append((f"# Bedrooms: {bedrooms}", features["bedrooms"]))
    ordered_features.append((f"# Bathrooms: {bathrooms}", features["bathrooms"]))

    # then the categorical variables
    for i in [
        ("Parking", "parking"),
        ("Housing Type", "housing_type"),
        ("Laundry", "laundry"),
    ]:
        cased_label, data_label = i
        if features[data_label] != 0:
            ordered_features.append(
                (f"{cased_label}: {form_data[data_label]}", features[data_label])
            )

    # then the pet status
    cats, dogs = form_data["cats_ok"], form_data["dogs_ok"]
    pet_status = display_pet_status(cats, dogs)
    ordered_features.append((pet_status, features["cats_ok"] + features["dogs_ok"]))

    # then the remaining booleans
    for i in [
        ("no_smoking", "Smoke-Free", "Smoke-Friendly"),
        ("is_furnished", "Furnished", "Not Furnished"),
        ("wheelchair_acccess", "Wheelchair-Friendly", "No Wheelchair Access"),
        ("ev_charging", "Electronic Vehicle Charging", "No Electric Vehicle Charging"),
    ]:
        data_label, truthy, falsy = i
        if features[data_label] != 0:
            if form_data[data_label]:
                ui_label = truthy
            else:
                ui_label = falsy
        ordered_features.append((ui_label, features[data_label]))

    return ordered_features


def clean_and_style(bias, contributions, form_data):
    """This takes the results of a ti.predict() call and converts them into a
    user - friendly HTML table, showing the contribution of each feature"""
    cleaned_features = clean_features(contributions, model_columns)
    ordered_features = order_features(bias, cleaned_features, form_data)
    return ordered_features


def process_form(model, form_data):
    """Accepts an machine learning model and a set of user-inputted observations
    about an apartment.

    Returns a tuple of a) the model's prediction for that apartment's price, and
    b) an HTML table summarizing the treeinterpreter feature contributions for
    that prediction."""
    print("step 1 - ingest form data")
    dataframe = form_data_to_dataframe(form_data)
    print("step 2 - ML")
    estimate, bias, contributions = predict_and_unpack(model, dataframe)
    print("step 3 - Clean and style")
    explanation = clean_and_style(bias, contributions, form_data)
    print("step 4 - Ready to go")
    return round(estimate, 2), explanation


@app.route("/", methods=["GET", "POST"])
def home():
    form = EstimateForm(request.form)
    estimate, explanation = None, None
    if request.method == "POST" and form.validate():
        estimate, explanation = process_form(
            model,
            form.data,  # pylint: disable=no-member
        )
    return render_template(
        "form.html",
        form=form,
        estimate=estimate,
        explanation=explanation,
    )


if __name__ == "__main__":
    app.run(PORT=5000)
