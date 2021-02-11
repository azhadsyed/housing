import json
import pickle

from flask import Flask
import joblib
import shap
from geopy.geocoders import Nominatim

from ml.train import RandomForestModel

with open(".data/options.json", "r") as f:
    options = json.load(f)

app = Flask(__name__)

app.model_columns = options["column order"]
app.categorical_features = options["categorical features"]

rfm = RandomForestModel()
rfm.train_random_forest(".data/data.csv", "price", ["id"])

app.model = rfm.model
app.feature_names = app.model.steps[0][1].get_feature_names()
app.explainer = shap.TreeExplainer(app.model.steps[1][1])

app.nominatim = Nominatim(user_agent="nyc_rent_estimator")

app.ti_enabled = False

from app import views