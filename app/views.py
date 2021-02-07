from app import app
from app.forms import EstimateForm
from app.utils import process_form
from flask import request, render_template


@app.route("/", methods=["GET", "POST"])
def home():
    form = EstimateForm(request.form)
    estimate, explanation = None, None
    if request.method == "POST" and form.validate():
        estimate, explanation = process_form(
            app.model,
            form.data,  # pylint: disable=no-member
        )
    return render_template(
        "form.html",
        form=form,
        estimate=estimate,
        explanation=explanation,
    )
