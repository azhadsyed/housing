"""
Rent Prediction CLI to smoke-test cached models.

Accepts a JSON file in this format:

{
    bedrooms: 2,
    bathrooms: 2,
    housing_type: apartment
    ...
}

returns a floating precision recommendation for how much the apartment should
be listed.
"""
from joblib import load
import sys
import json
import pandas as pd

path = sys.argv[1]
with open(path, "r") as f:
    request = json.load(f)
request = pd.DataFrame.from_dict({k: [v] for k, v in request.items()})

model = load("model.joblib")
print(model.predict(request))