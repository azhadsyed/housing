# warm for the winter - Middle School, Aso

import json

import pandas as pd
from flask import Flask, render_template, request, Markup
from geopy.geocoders import Nominatim
from joblib import load
from treeinterpreter import treeinterpreter as ti

from .forms import EstimateForm

# from here until first function definition, this could all live in some sort of config?

with open("options.json", "r") as f:
    options = json.load(f)
column_order = options["column order"]
categorical_features = options["categorical features"]

app = Flask(__name__)
app.config.from_pyfile("config.py")

model = load("model.joblib")
feature_names = model.steps[0][1].get_feature_names()

geolocator = Nominatim(user_agent="nyc_rent_estimator")


def form_data_to_dataframe(form_data):
    if "submit" in form_data.keys():
        del form_data["submit"]
    location = geolocator.geocode(form_data["address"])
    form_data["latitude"] = location.latitude
    form_data["longitude"] = location.longitude
    dict_for_pandas = {k: [v] for k, v in form_data.items()}
    dataframe = pd.DataFrame.from_dict(dict_for_pandas)
    return dataframe[column_order]


def predict_and_unpack(model, dataframe) -> (float, float, list):
    """Accepts an ML model and a dataframe with one row. Returns a tuple with
    one floating precision estimate, one floating precision bias, and one two-
    dimensional array of feature names and floating precision contributions."""
    if len(dataframe) != 1:
        raise ValueError("dataframe cannot have more than one row.")
    estimate, bias, contributions = ti.predict(
        model.steps[-1][1],  # model itself -  random forest
        model.steps[0][1].transform(dataframe),  # transformed observation
    )
    estimate = estimate[0, 0]
    bias, contributions = bias[0], contributions[0]
    contributions = list(filter(lambda x: x[1] != 0, zip(feature_names, contributions)))
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


def order_features(bias, features, bedrooms, bathrooms) -> list:
    """Accepts a clean dictionary of features. Returns a UI-friendly array of
    tuples (ordered) explaining feature contributions."""
    ordered_features = []
    ordered_features.append(
        (
            f"Avg. {bedrooms}br {bathrooms}ba in NYC",
            bias + features["bedrooms"] + features["bathrooms"],
        )
    )
    ordered_features.append(("location", features["latitude"] + features["longitude"]))

    for k, v in features.items():
        if k not in ["bedrooms", "bathrooms", "longitude", "latitude"]:
            if v != 0:
                ordered_features.append((k, v))
    return ordered_features


def clean_and_style(bias, contributions, bedrooms, bathrooms):
    """This takes the results of a ti.predict() call and converts them into a
    user - friendly HTML table, showing the contribution of each feature"""
    cleaned_features = clean_features(contributions, column_order)
    ordered_features = order_features(bias, cleaned_features, bedrooms, bathrooms)
    contribution_tags = []
    for i in ordered_features:
        feature, contribution = i
        contribution_tags.append(
            "<tr><td>"
            + feature
            + "</td><td>"
            + str(round(contribution, 2))
            + "</td></tr>"
        )
    return Markup("<table>" + "".join(contribution_tags) + "</table>")


def process_form(model, form_data):
    """Accepts an machine learning model and a set of user-inputted observations
    about an apartment.

    Returns a tuple of a) the model's prediction for that apartment's price, and
    b) an HTML table summarizing the treeinterpreter feature contributions for
    that prediction."""
    dataframe = form_data_to_dataframe(form_data)
    estimate, bias, contributions = predict_and_unpack(model, dataframe)
    explanation = clean_and_style(
        bias, contributions, form_data["bedrooms"], form_data["bathrooms"]
    )
    return round(estimate, 2), explanation


@app.route("/", methods=["GET", "POST"])
def home():
    form = EstimateForm(request.form)
    estimate, explanation = None, None
    if request.method == "POST" and form.validate():
        estimate, explanation = process_form(
            model, form.data  # pylint: disable=no-member
        )
    return render_template(
        "form.html",
        form=form,
        estimate=estimate,
        explanation=explanation,
    )


if __name__ == "__main__":
    app.run(PORT=5000)
