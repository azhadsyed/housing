# warm for the winter - Middle School, Aso

import json

import pandas as pd
from flask import Flask, render_template, request, Markup
from joblib import load
from treeinterpreter import treeinterpreter as ti

from forms import EstimateForm

with open("options.json", "r") as f:
    options = json.load(f)
column_order = options["column order"]  # I'm okay with this being a global

app = Flask(__name__)
app.config.from_pyfile("config.py")

# what if there are more models in circulation? Not sure this design will scale
model = load("model.joblib")  # treat this like a black box, assume it works for now
feature_names = model.steps[0][
    1
].get_feature_names()  # parse and clean the feature names for downstream use


def clean_features(array, order):
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


def order_features(bias, features):
    """Accepts a clean dictionary of features. Returns a UI-friendly array of
    tuples (ordered) explaining feature contributions."""
    ordered_features = []
    ordered_features.append(
        ("base", bias + features["bedrooms"] + features["bathrooms"])
    )
    for k, v in features.items():
        if k not in ["bedrooms", "bathrooms"]:
            ordered_features.append((k, v))
    return ordered_features


def predict_and_unpack(model, dataframe):
    """This is a helper function to do some much needed unpacking of the values
    returned by treeinterpreter.predict()."""
    estimate, bias, contributions = ti.predict(
        model.steps[-1][1],  # model itself -  random forest
        model.steps[0][1].transform(dataframe),  # transformed observation
    )
    estimate = round(estimate[0, 0], 2)
    bias, contributions = bias[0], contributions[0]
    return estimate, bias, contributions


def clean_and_style(bias, contributions):
    """This takes the results of a ti.predict() call and converts them into a
    user - friendly HTML table, showing the contribution of each feature"""
    feature_explanations = list(
        filter(lambda x: x[1] != 0, zip(feature_names, contributions))
    )
    cleaned_features = clean_features(feature_explanations, column_order)
    ordered_features = order_features(bias, cleaned_features)
    contribution_tags = []  # is there some kind of map/reduce to do this?
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


def process_form(model, data):
    """Accepts an machine learning model and a set of user-inputted observations
    about an apartment.

    Returns a tuple of a) the model's prediction for that apartment's price, and
    b) an HTML table summarizing the treeinterpreter feature contributions for
    that prediction."""
    dataframe = pd.DataFrame.from_dict({k: [v] for k, v in data.items()})[column_order]
    estimate, bias, contributions = predict_and_unpack(model, dataframe)
    explanation = clean_and_style(bias, contributions)
    return estimate, explanation


# refactor - with classes - 10 lines, 316 characters
@app.route("/", methods=["GET", "POST"])
def home():
    form = EstimateForm(request.form)
    estimate, explanation = None, None
    if request.method == "POST" and form.validate():
        # submit gets thrown into the JSON mix by WTForms,
        # so delete it and we can use form.data as is!
        data = form.data
        del data["submit"]
        estimate, explanation = process_form(model, data)
    breakpoint()
    return render_template(
        "form.html",
        form=form,
        estimate=estimate,  # s/b None or a Decimal
        explanation=explanation,  # s/b None or an HTML table element
    )


if __name__ == "__main__":
    app.run(PORT=5000)
