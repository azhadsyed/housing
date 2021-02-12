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


with open(".data/options.json", "r") as f:
    options = json.load(f)

app = Flask(__name__)

from io import BytesIO

print("connect")
s3 = boto3.resource("s3", aws_access_key_id=key_id, aws_secret_access_key=secret_key)
bucket_str = "nycestimator"
bucket_key = "model.joblib"
with BytesIO() as data:
    print("download")
    s3.Bucket(bucket_str).download_fileobj(bucket_key, data)
    data.seek(0)
    print("load into RAM")
    app.model = load(data)

app.model_columns = options["column order"]
app.categorical_features = options["categorical features"]
app.feature_names = app.model.steps[0][1].get_feature_names()
app.explainer = shap.TreeExplainer(app.model.steps[1][1])
app.nominatim = Nominatim(user_agent="nyc_rent_estimator")
app.ti_enabled = False

from app import views