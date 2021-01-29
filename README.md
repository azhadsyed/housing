# NYC Rent Estimator App

Codebase for my NYC rent estimator app. Accepts an apartment's location, size, and amenities as inputs; outputs a listing price estimate and an ELI5 explanation of how that listing price is calculated.

Built in Python using Flask/Jinja, Pandas, Pymongo, Scikit-learn, and other community packages.

## TODO

Database Deployment (6)

- [x] write script to extract/load craigslist data to local MongoDB
- [x] setup MongoDB on droplet
- [x] clone repo and setup venv on droplet
- [x] test "extract_load.py" on the droplet
- [x] setup cron job to execute "extract_load.py newyork apa" daily
- [x] trouble shoot the cron job and make sure its working.

Machine Learning (3)

- [x] write script to transform raw data into feature vector (saves vector to CSV)
- [x] write training script to regress model onto data.csv (scrapped xgboost)
- [x] write script to train Scikit-learn GBR model on feature vector (saves model to .model)

Backend (2)

- [x] write a CLI that accepts a sample and returns a price estimate
- [x] write flask api endpoint that accepts parameters in body and returns a price estimate

Frontend

- [x] Wireframe the UX
- [x] Static UI layout
- [x] feed the categories of "laundry" and "parking" through to the front-end
- [x] implement flask-wtf (forms) prediction
- [x] setup eli5 display
- [x] write frontend unit tests
- [x] add support for location as a model variable
- [x] Style app (basics)
- [x] Change explanation details

Quick Fixes (from pair session w/Annie)

- [x] estimate button
- [x] "Your Estimate" Font size
- [o] input fonts (not super relevant)

Staging

- [ ] Write mock API for GeoPy calls
- [ ] Come up with a way to test the app on a copy of production data
- [ ] Assess results and fix bugs
- [ ] Setup a CircleCI pipeline

Deployment

- [ ] setup cron job to update CSV and model daily
- [ ] Containerize the application
- [ ] Deploy using DigitalOcean

Implementation Fixes

- [ ] Refactor using App Factory and some functional tests

A/B Testing different algorithms

- [ ] Refactor, and write tests for, the ML pipeline
- [ ] Linear Regression, KNN, Decision Tree, Random Forest, GBR, XGBoost, Catboost
- [ ] Geospatial Regression Algorithms
- [ ] Build a shootout results view that goes into the math/statistics
- [ ] Incorporate the winner

Data Quality

- [ ] Setup bad-data filters
- [ ] Automate once-weekly checks to remove flagged/removed/expired listings

Frontend Updates

- [ ] add hover tips explaining what each variable means (see design of [SmartAsset Calculator](https://smartasset.com/taxes/income-taxes))
- [ ] Article: Interesting Things I learned about rent in NYC

Mean Encoding

- [ ] Implement mean-target encoding with k-fold regularization for cat. variables

Other

- [ ] Visualize median price by zipcode in NYC
- [ ] How different is a zipcode from the average of all the other touching zipcodes?
