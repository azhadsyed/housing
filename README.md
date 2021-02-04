# NYC Rent Estimator App

Codebase for my NYC rent estimator app. Accepts an apartment's location, size, and amenities as inputs; outputs a listing price estimate and an ELI5 explanation of how that listing price is calculated.

Built in Python using Flask/Jinja, Pandas, Pymongo, Scikit-learn, and other community packages.

## Roadmap

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
- [x] Style app (basic)
- [x] Change explanation details

Quick Fixes (from pair session w/Annie)

- [x] estimate button
- [x] "Your Estimate" Font size

Staging

- [x] Write mock API for GeoPy calls
- [x] Come up with a way to test the app on a copy of production data (read only)
- [x] Assess results and fix bugs
- [x] Cache API calls (flask-caching)
- [x] Implement SHAP and see how it looks (not really more interpretable, but faster)
- [x] Write a test that asserts that the page loads, and doesn't break when you estimate on a test case

CI/CD

- [ ] setup cron job to update CSV and model daily
- [ ] Setup a CircleCI pipeline
- [ ] Deploy using DigitalOcean and Docker

Post-Launch Fixes

- [ ] change "input area" fonts to Courier (not super important)

Data Quality

- [ ] Setup bad-data filters
- [ ] Automate once-weekly checks to remove flagged/removed/expired listings

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
- [ ] Article: Interesting Things I learned about rent in NYC
- [ ] Visualize median price by zipcode in NYC
- [ ] How different is a zipcode from the average of all the other touching zipcodes?
