from wtforms import (
    BooleanField,
    Form,
    IntegerField,
    DecimalField,
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
    address = StringField(
        "Street Address",
        [InputRequired()],
        render_kw={"size": 30, "font-family": "Courier"},
    )
    bedrooms = IntegerField(
        "No. Bedrooms",
        [InputRequired()],
        render_kw={"size": 5, "font-family": "Courier"},
    )
    bathrooms = DecimalField(
        "No. Bathrooms",
        [InputRequired()],
        render_kw={"size": 5, "font-family": "Courier"},
    )
    housing_type = SelectField("Housing Type", choices=housing_type_choices)
    laundry = SelectField("Laundry", choices=laundry_choices)
    parking = SelectField("Parking", choices=parking_choices)
    is_furnished = BooleanField("Furnished")
    no_smoking = BooleanField("No Smoking")
    wheelchair_acccess = BooleanField("Wheelchair Access")
    ev_charging = BooleanField("Electric Vehicle Charging")
    cats_ok = BooleanField("Cats Allowed")
    dogs_ok = BooleanField("Dogs Allowed")
    submit = SubmitField(
        "Estimate",
        render_kw={
            "style": "font-size:120%;background-color:black;border:none;color:white;font-family:Courier;border-radius:4px"
        },
    )
