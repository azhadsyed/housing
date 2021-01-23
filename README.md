# NYC Rent Estimator App

Codebase for my NYC rent estimator app. Accepts an apartment's location, size, and amenities as inputs; outputs a listing price estimate and an ELI5 explanation of how that listing price is calculated.

Built in Python using Flask/Jinja, Pandas, Pymongo and Scikit-learn, and other community packages.

## FAQ

**Is this the same as Rentestimator.com?** Sort of.

1. Rentestimator.com isn't free after the first 5 searches.
2. Rentestimator.com doesn't factor in laundry, parking, and pet-status, and hasn't scaled to other features recently
3. Rentestimator.com uses a k-nearest-neighbors approach, I use a decision-tree approach which is more robust to outliers, allows for learning to transfer between different areas of NYC where eligible, and is robust to interaction effects (for example, in certain areas of NYC are in-unit washers/dryers given more of a premium than in other areas?)

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
- [ ] get eli5 endpoint working

Frontend

- [x] Wireframe the UX
- [x] Static UI layout
- [x] feed the categories of "laundry" and "parking" through to the front-end
- [x] implement flask-wtf (forms) prediction
- [ ] setup eli5 display
- [ ] Style

Deployment

- [ ] setup cron job to update CSV and model daily
- [ ] Containerize the application
- [ ] Deploy using DigitalOcean

Mean Encoding

- [ ] Implement mean-target encoding with k-fold regularization for cat. variables

Google Maps

- [ ] Incorporate Google Maps Geocode data into the pipeline
- [ ] Filter out listings outside of NYC
- [ ] Handle zip codes that aren't in the training data

A/B Testing different algorithms

- [ ] Linear Regression, KNN, Decision Tree, Random Forest, GBR, XGBoost, Catboost
- [ ] Geospatial Regression Algorithms
- [ ] Build a shootout results view that goes into the math/statistic
- [ ] Incorporate the winner

Other

- [ ] Visualize median price by zipcode in NYC
- [ ] How different is a zipcode from the average of all the other touching zipcodes?
