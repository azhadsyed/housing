from wtforms import (
    BooleanField,
    Form,
    IntegerField,
    SelectField,
    StringField,
    SubmitField,
)
from wtforms.validators import InputRequired, ValidationError

from .config import options

laundry_choices = options["laundry"]
parking_choices = options["parking"]
housing_type_choices = options["housing type"]


class EstimateForm(Form):
    address = StringField("Street Address", [InputRequired()])
    bedrooms = IntegerField("No. Bedrooms", [InputRequired()])
    bathrooms = IntegerField("No. Bathrooms", [InputRequired()])
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
