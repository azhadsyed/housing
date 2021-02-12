import os
import tempfile
import boto3
from dotenv import load_dotenv

from train import RandomForestModel
from joblib import dump

if os.path.exists(".env"):
    load_dotenv()

key_id = os.environ["AWS_ACCESS_KEY_ID"]
secret_key = os.environ["AWS_SECRET_KEY"]

s3 = boto3.resource("s3", aws_access_key_id=key_id, aws_secret_access_key=secret_key)
rfm = RandomForestModel()
rfm.train_random_forest(".data/data.csv", "price", ["id"])

# Cache the model to S3 for downstream use
model_filename = (
    "model_dev.joblib" if os.environ["FLASK_ENV"] == "development" else "model.joblib"
)

with tempfile.TemporaryFile() as fp:
    dump(rfm.model, fp)
    fp.seek(0)
    s3.Bucket("nycestimator").put_object(Key=model_filename, Body=fp.read())