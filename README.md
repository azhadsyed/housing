# NYC Rent Estimator App

poetry export -f requirements.txt --output requirements.txt --without-hashes

Codebase for my NYC rent estimator app. Accepts an apartment's location, size, and amenities as inputs; outputs an estimated listing price and a SHAP-driven explanation of how that listing price is calculated.

Built in Python using Flask/Jinja, Pandas, Pymongo, Scikit-learn, and other community packages.

Deployed using CircleCI and Docker.

## CI/CD Brainstorming

What does this build need? Think infrastructure as code

1. tests need to pass
   - On merge
   - On build/deploy
2. Python dependencies
   - on deploy, make sure they're all there
3. Data Dependencies: data.csv, model.joblib, options.json
   - They get updated daily using python scripts
   - The app cannot run without them
4. Connection to the MongoDB server
5. The MongoDB itself
   - Decoupled from the docker container, accessed using a connection_string
   - It gets updated daily using a python script

## TODO

Hotfixes

- [ ] ensure all searches are local to NYC
- [ ] differentiate between area search and street search

CircleCI

- [ ] Write a CircleCI Pipeline that satisfies build/testing criteria

Data Quality

- [ ] Setup bad-data filters
- [ ] Automate once-weekly checks to remove flagged/removed/expired listings

Data Exploration

- [ ] Shap interaction reporting
- [ ] Building NYC Variable maps
- [ ] Article: Interesting Things I learned about rent in NYC
- [ ] Visualize median price by zipcode in NYC
- [ ] How different is a zipcode from the average of all the other touching zipcodes?

A/B Testing different algorithms

- [ ] Implement mean-target encoding with k-fold regularization for cat. variables
- [ ] Refactor, and write tests for, the ML pipeline
- [x] Linear Regression, KNN, Decision Tree, Random Forest
- [ ] GBR, XGBoost, Catboost
- [ ] Geospatial Regression Algorithms
- [ ] Build a shootout results view that goes into the math/statistics
- [ ] Incorporate the winner

Frontend Updates

- [ ] add hover tips explaining what each variable means (see design of [SmartAsset Calculator](https://smartasset.com/taxes/income-taxes))
