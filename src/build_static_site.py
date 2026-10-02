"""Build a static copy of the website (no Python server needed) for hosting on Netlify.

Run from the project folder after training:  python src/build_static_site.py
Creates the site/ folder:
    site/index.html             the same page Flask shows
    site/static/                style.css, script.js, predict_local.js
    site/static/model.json      the trained model's numbers (weights and intercept)

Why this works: a trained Linear Regression model is just numbers.
    log(price) = intercept + weight1 x value1 + weight2 x value2 + ...
So we can copy those numbers into a JSON file and do the same sum in JavaScript.
"""
import json
import shutil

from app import CAR_NAMES, NUMBER_LIMITS, PROJECT_DIR, app, model
from preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

SITE_DIR = PROJECT_DIR / "site"


def export_model():
    """Pull every number the prediction needs out of the saved Pipeline."""
    preprocess = model.named_steps["preprocess"]
    regression = model.named_steps["linear_regression"].regressor_  # the LinearRegression inside

    encoder = preprocess.named_transformers_["categories"].named_steps["one_hot"]

    # The coefficients are in the same order as the preprocessed columns:
    # first the 5 number columns, then one column per category value.
    # (Missing-value filling is not exported: the form requires every field.)
    weights = iter(regression.coef_)
    numeric = {column: next(weights) for column in NUMERIC_COLUMNS}
    categorical = {
        column: {value: next(weights) for value in values}
        for column, values in zip(CATEGORICAL_COLUMNS, encoder.categories_)
    }

    return {
        "intercept": regression.intercept_,
        "numeric": numeric,
        "categorical": categorical,
        "car_names": CAR_NAMES,
        "limits": NUMBER_LIMITS,
    }


def main():
    shutil.rmtree(SITE_DIR, ignore_errors=True)
    shutil.copytree(PROJECT_DIR / "static", SITE_DIR / "static")

    # Render the page exactly as Flask would, then load predict_local.js before script.js.
    # script.js sees window.predictLocally and uses it instead of calling Flask's /predict.
    html = app.test_client().get("/").get_data(as_text=True)
    local_script = '<script src="/static/predict_local.js"></script>\n  <script src="/static/script.js">'
    html = html.replace('<script src="/static/script.js">', local_script)
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")

    model_json = json.dumps(export_model(), indent=1, default=float)  # default=float turns NumPy numbers into plain numbers
    (SITE_DIR / "static" / "model.json").write_text(model_json, encoding="utf-8")

    print(f"Static site written to {SITE_DIR.relative_to(PROJECT_DIR)}/")


if __name__ == "__main__":
    main()
