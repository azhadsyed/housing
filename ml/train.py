"""
train.py receives a training dataset (data.csv) that contains:
1. numeric and categorical values
2. no missing values

With this dataset, it trains a model that predicts the price based 
on the feature vector. The model's accuracy gets printed when its trained

The preprocessing pipeline follows the design principles laid out here:
https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
"""

import json
import os
import pickle
from statistics import mean

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_columns", None)

# 0. Read the data
data = pd.read_csv("data/data.csv")

# ["housing_type", "laundry", "parking"]
categorical_features = data.dtypes[data.dtypes == "object"].index.values

# 1. Split the data
X = data.drop(["price", "id"], axis=1, inplace=False)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. preprocess the features for training
ct = make_column_transformer(
    (
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features,
    ),
    remainder="passthrough",
)
rf = RandomForestRegressor()

# 3. Fit and score the model
model = make_pipeline(ct, rf)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
error = abs(predictions - y_test)
print(error.describe())

# 4. Cache the model for downstream use
dump(model, open("data/model.joblib", "wb"))