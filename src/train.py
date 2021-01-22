"""
train.py receives a training dataset (train.csv) that contains:
1. numeric and categorical values
2. no missing values

With this dataset, it trains a model that predicts the price based 
on the feature vector. The model's accuracy gets logged when its trained,
but isn't called to predict here, that happens in predict.py

The preprocessing pipeline follows the design principles laid out here:
https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
"""

from statistics import mean

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import json, os

np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_columns", None)

# 0. Read the data
data = pd.read_csv("data.csv")
categorical_features = data.dtypes[
    data.dtypes == "object"
].index.values  # ["housing_type", "laundry", "parking"]

# 1. Split the data
X = data.drop(["price", "id"], axis=1, inplace=False)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. preprocess the features for training
ct = make_column_transformer(
    (
        OneHotEncoder(),
        categorical_features,
    ),
    remainder="passthrough",
)
rf = RandomForestRegressor()

# 3. Fit and score the model
model = make_pipeline(ct, rf)
model.fit(X_train, y_train)

data["predictions"] = model.predict(X)
data["error"] = abs(data["predictions"] - data["price"])
print(data["error"].describe())

# 3.1 Testing ELI5
test = {
    "cats_ok": True,
    "dogs_ok": False,
    "housing_type": "apartment",
    "laundry": "laundry in bldg",
    "bedrooms": 3,
    "bathrooms": 1,
    "parking": "street parking",
    "no_smoking": False,
    "is_furnished": False,
    "wheelchair_acccess": True,
    "ev_charging": False,
}

testX = pd.DataFrame.from_dict({k: [v] for k, v in test.items()})

from treeinterpreter import treeinterpreter as ti

prediction, bias, contributions = ti.predict(
    model.steps[-1][1], model.steps[0][1].transform(testX)
)
print(contributions)  # feature contributions
print(bias)  # feature contributions


# 4. Cache the model and list of categorical features for downstream use
dump(model, "model.joblib")