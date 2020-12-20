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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_columns", None)

# 0. Read the data
data = pd.read_csv("data.csv")
categorical_features = data.dtypes[data.dtypes == "object"].index.values

# 1. Split the data
X = data.drop(["price", "id"], axis=1, inplace=False)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. preprocess the features for training
ct = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="passthrough",
)
gbr = GradientBoostingRegressor()

# 3. Fit and score the model
model = make_pipeline(ct, gbr)
model.fit(X_train, y_train)

data["predictions"] = model.predict(X)
data["error"] = abs(data["predictions"] - data["price"])
print(data.sort_values(by="error", ascending=False).head(20))

# 4. Cache the model for downstream use
dump(gbr, "model.joblib")