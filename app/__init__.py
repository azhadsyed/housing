import json
import pickle
from dotenv import load_dotenv
import os
import tempfile

from flask import Flask
import boto3
import shap
from geopy.geocoders import Nominatim
from joblib import load

if os.path.exists(".env"):
    load_dotenv()

key_id = os.environ["AWS_ACCESS_KEY_ID"]
secret_key = os.environ["AWS_SECRET_KEY"]

s3 = boto3.resource("s3", aws_access_key_id=key_id, aws_secret_access_key=secret_key)

with open(".data/options.json", "r") as f:
    options = json.load(f)

app = Flask(__name__)

with tempfile.TemporaryFile() as fp:
    s3.download_fileobj("nycestimator", "model.joblib", fp)
    app.model = load(fp)

app.model_columns = options["column order"]
app.categorical_features = options["categorical features"]
app.feature_names = app.model.steps[0][1].get_feature_names()
app.explainer = shap.TreeExplainer(app.model.steps[1][1])
app.nominatim = Nominatim(user_agent="nyc_rent_estimator")
app.ti_enabled = False

from app import views