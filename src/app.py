from flask import Flask, request
from joblib import load
import pandas as pd

app = Flask(__name__)
model = load("model.joblib")


def predictors_from_request(json):
    return pd.DataFrame.from_dict({k: [v] for k, v in json.items()})


@app.route("/predict", methods=["POST"])
def predict():
    predictors = predictors_from_request(request.get_json())
    prediction = model.predict(predictors)[0]
    return {"prediction": round(prediction, 2)}


if __name__ == "__main__":
    app.run(debug=True, port=5000)
