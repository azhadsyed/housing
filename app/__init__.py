import json
import pickle

from flask import Flask
import joblib
import shap
from geopy.geocoders import Nominatim

with open("data/options.json", "r") as f:
    options = json.load(f)

app = Flask(__name__)
app.config.from_pyfile("config.py")

app.model_columns = options["column order"]
app.categorical_features = options["categorical features"]

app.model = joblib.load("data/model.joblib")
app.feature_names = app.model.steps[0][1].get_feature_names()
app.explainer = shap.TreeExplainer(app.model.steps[1][1])

app.nominatim = Nominatim(user_agent="nyc_rent_estimator")
app.all_hands_on_deck = pickle.load(open("data/geopy_mock_response.pkl", "rb"))

app.ti_enabled = False

from app import views