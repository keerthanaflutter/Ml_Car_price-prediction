"""Train the used-car price model from start to finish and save it.

Run from the project folder:  python src/train_model.py
"""
from pathlib import Path

import joblib
import pandas as pd

from evaluation import compare_prices, evaluate_model
from linear_regression import train_model
from preprocessing import clean_data, inspect_data, load_data, split_data

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "used_cars.csv"
MODEL_PATH = PROJECT_DIR / "models" / "car_price_model.pkl"


def rupees(amount):
    return f"Rs {amount:,.0f}"


def main():
    print("1. Load the dataset")
    cars = load_data(DATA_PATH)

    print("\n2. Inspect the data")
    inspect_data(cars)

    print("\n3. Clean the data")
    cars = clean_data(cars)
    print(f"   {len(cars)} cars have a price and are kept")
    print("   Other missing values are filled inside the pipeline (median / most common value)")

    print("\n4. Split into training and testing data")
    X_train, X_test, y_train, y_test = split_data(cars)
    print(f"   Training cars: {len(X_train)}   Testing cars: {len(X_test)}")

    print("\n5. Train the pipeline (preprocessing + one-hot encoding + Linear Regression)")
    model = train_model(X_train, y_train)
    weights = model.named_steps["linear_regression"].regressor_.coef_
    print(f"   Learned {len(weights)} weights + intercept")

    print("\n6. Predict prices for the test cars")
    y_pred = model.predict(X_test)

    print("\n7. Evaluate the model")
    for name, value in evaluate_model(y_test, y_pred).items():
        if name == "R2":
            print(f"   {name:5}: {value:.3f}")
        elif name == "MSE":
            print(f"   {name:5}: {value:,.0f}  (squared rupees, so read RMSE instead)")
        else:
            print(f"   {name:5}: {rupees(value)}")

    print("\n   Actual vs predicted (first 10 test cars):")
    table = compare_prices(y_test, y_pred)
    details = X_test[["brand", "model", "year", "km_driven"]]
    print(details.join(table).head(10).to_string(index=False))

    print("\n8. Try the model on a new car")
    new_car = pd.DataFrame([{
        "brand": "Hyundai", "model": "Creta", "year": 2019, "km_driven": 45000,
        "fuel_type": "Diesel", "transmission": "Manual", "engine_cc": 1493,
        "mileage_kmpl": 21.4, "owners": 1, "seller_type": "Dealer",
    }])
    print(f"   2019 Hyundai Creta, 45,000 km -> {rupees(model.predict(new_car)[0])}")

    print("\n9. Save the trained model")
    MODEL_PATH.parent.mkdir(exist_ok=True)  # create models/ if it does not exist
    joblib.dump(model, MODEL_PATH)
    print(f"   Saved to {MODEL_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
