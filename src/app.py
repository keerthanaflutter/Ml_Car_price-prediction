"""Flask website: shows the form and answers price predictions with the trained model.

Train the model first (python src/train_model.py), then run:  python src/app.py
"""
import os
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from preprocessing import CATEGORICAL_COLUMNS, load_data

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "car_price_model.pkl"

# templates/ and static/ live in the project folder, one level above src/.
app = Flask(__name__, template_folder=PROJECT_DIR / "templates", static_folder=PROJECT_DIR / "static")

if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Train it first:  python src/train_model.py")
model = joblib.load(MODEL_PATH)  # loaded once at startup, reused for every prediction

# Dropdown choices = the categories the model learned during training (read from its OneHotEncoder).
encoder = model.named_steps["preprocess"].named_transformers_["categories"].named_steps["one_hot"]
CHOICES = {column: list(values) for column, values in zip(CATEGORICAL_COLUMNS, encoder.categories_)}

# "Car Name" is one dropdown of car models grouped by brand, e.g. Hyundai -> Creta, i20, Verna.
cars = load_data(PROJECT_DIR / "data" / "used_cars.csv")
CAR_NAMES = cars.groupby("brand")["model"].unique().apply(sorted).to_dict()
MODEL_TO_BRAND = {car_model: brand for brand, models in CAR_NAMES.items() for car_model in models}

# Allowed (min, max) for each number on the form. The model cannot give sensible prices far outside these.
NUMBER_LIMITS = {
    "year": (1990, date.today().year),
    "km_driven": (0, 1_000_000),
    "engine_cc": (600, 7000),
    "mileage_kmpl": (5, 50),
    "owners": (1, 4),
}


def validate(form):
    """Check the submitted car details. Returns (car, None) if valid, or (None, error message)."""
    car = {}

    car_model = form.get("model")
    if car_model not in MODEL_TO_BRAND:
        return None, "Please choose a car name from the list."
    car["model"] = car_model
    car["brand"] = MODEL_TO_BRAND[car_model]

    for column in ["fuel_type", "transmission", "seller_type"]:
        if form.get(column) not in CHOICES[column]:
            return None, f"Please choose a valid {column.replace('_', ' ')}."
        car[column] = form[column]

    for column, (low, high) in NUMBER_LIMITS.items():
        try:
            value = float(form.get(column))
        except (TypeError, ValueError):
            return None, f"{column.replace('_', ' ').capitalize()} must be a number."
        if not low <= value <= high:
            return None, f"{column.replace('_', ' ').capitalize()} must be between {low:,} and {high:,}."
        car[column] = value

    return car, None


@app.get("/")
def home():
    return render_template("index.html", car_names=CAR_NAMES, choices=CHOICES, limits=NUMBER_LIMITS)


@app.post("/predict")
def predict():
    """Receive car details as JSON, return the estimated price as JSON."""
    car, error = validate(request.get_json(silent=True) or {})
    if error:
        return jsonify({"error": error}), 400

    # The pipeline does all preprocessing (fill gaps, one-hot encode) and returns a price in rupees.
    price = model.predict(pd.DataFrame([car]))[0]
    return jsonify({"price": round(float(price))})


if __name__ == "__main__":
    # Default port 5000. On a Mac where AirPlay Receiver uses 5000, run:  PORT=5001 python src/app.py
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
