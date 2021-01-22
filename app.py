# warm for the winter - Middle School, Aso

import json

import pandas as pd
from flask import Flask, render_template, request, Markup
from joblib import load
from treeinterpreter import treeinterpreter as ti

from forms import EstimateForm
from src.parse import clean_features, order_features

model = load("model.joblib")

# parse and clean the feature names for downstream use
features = model.steps[0][1].get_feature_names()


with open("options.json", "r") as f:
    options = json.load(f)
column_order = options["column order"]


app = Flask(__name__)
app.config.from_pyfile("config.py")


def predictors_from_request(json):
    return pd.DataFrame.from_dict({k: [v] for k, v in json.items()})


@app.route("/api/predict", methods=["POST"])
def predict():
    if not request.get_json():
        return {"error": "did not receive JSON in the body"}
    predictors = predictors_from_request(request.data)
    prediction = model.predict(predictors)[0]
    return {"prediction": round(prediction, 2)}


@app.route("/", methods=["GET", "POST"])
def home():
    form = EstimateForm()
    estimate, contribution_element = None, None
    if request.method == "POST":
        # data preparation
        predictors = form.data  # json
        del predictors["submit"]
        del predictors["csrf_token"]
        predictor_df = predictors_from_request(predictors)
        predictor_df = predictor_df[column_order]

        # model prediction and explanation
        estimate, bias, contributions = ti.predict(
            model.steps[-1][1],
            model.steps[0][1].transform(predictor_df),
        )
        estimate = round(estimate[0, 0], 2)
        bias, contributions = bias[0], contributions[0]

        feature_explanations = list(
            filter(lambda x: x[1] != 0, zip(features, contributions))
        )
        print(bias, feature_explanations)
        # output preparation
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
        contribution_element = "<table>" + "".join(contribution_tags) + "</table>"

    return render_template(
        "form.html",
        form=form,
        estimate=estimate,
        contributions=Markup(contribution_element),
    )


if __name__ == "__main__":
    app.run(PORT=5000)
