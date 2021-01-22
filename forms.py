from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SelectField, BooleanField, SubmitField
from wtforms.validators import InputRequired

import json

with open("options.json", "r") as f:
    options = json.load(f)
laundry_choices = options["laundry"]
parking_choices = options["parking"]
housing_type_choices = options["housing type"]


class EstimateForm(FlaskForm):
    address = StringField("Street Address")
    bedrooms = StringField("No. Bedrooms", [InputRequired()])
    bathrooms = StringField("No. Bathrooms", [InputRequired()])
    housing_type = SelectField("Housing Type", choices=housing_type_choices)
    laundry = SelectField("Laundry", choices=laundry_choices)
    parking = SelectField("Parking", choices=parking_choices)
    is_furnished = BooleanField("Furnished?")
    no_smoking = BooleanField("No Smoking?")
    wheelchair_acccess = BooleanField("Wheelchair Access?")
    ev_charging = BooleanField("Electric Vehicle Charging?")
    cats_ok = BooleanField("Cats Allowed?")
    dogs_ok = BooleanField("Dogs Allowed?")
    submit = SubmitField("Estimate")
