"""Load, inspect and prepare the used-car data for the model."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "selling_price"
NUMERIC_COLUMNS = ["year", "km_driven", "engine_cc", "mileage_kmpl", "owners"]
CATEGORICAL_COLUMNS = ["brand", "model", "fuel_type", "transmission", "seller_type"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def load_data(path):
    """Read the CSV file into a table (DataFrame)."""
    return pd.read_csv(path)


def inspect_data(df):
    """Print a quick overview: size, first rows, column types, missing values and number ranges."""
    print(f"   {len(df)} cars, {df.shape[1]} columns")
    print(df.head().to_string(index=False))
    print("\n   Column types:")
    print(df.dtypes.to_string())
    print("\n   Missing values per column:")
    print(df.isna().sum()[lambda counts: counts > 0].to_string())
    print("\n   Number summary:")
    print(df[NUMERIC_COLUMNS + [TARGET]].describe().round(1).to_string())


def clean_data(df):
    """Drop rows without a price: a car with no known price cannot teach the model anything.

    Other missing values are NOT filled here. The preprocessor below fills them,
    so the same rules are applied automatically when the website asks for a prediction.
    """
    return df.dropna(subset=[TARGET])


def build_preprocessor():
    """Describe how raw car details become numbers the model can learn from."""
    # Numbers: fill gaps with the median (middle value). The median is not
    # pulled up by a few very expensive luxury cars the way the mean would be.
    numeric_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
    ])

    # Categories: fill gaps with the most common value (e.g. "Petrol"), then one-hot encode.
    # One-hot encoding gives each category its own 0/1 column, because Linear Regression
    # only understands numbers. Example: fuel_type "Diesel" -> fuel_type_Diesel = 1, others = 0.
    # handle_unknown="ignore": a category never seen in training becomes all zeros instead of crashing.
    # sparse_output=False: return a normal table. With a sparse table, LinearRegression switches to
    # an approximate solver that handles unscaled columns like km_driven badly (R² fell to 0.83).
    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numbers", numeric_steps, NUMERIC_COLUMNS),
        ("categories", categorical_steps, CATEGORICAL_COLUMNS),
    ])


def split_data(df):
    """Split into inputs (X) and price (y), keeping 20% of cars aside for testing.

    The model never sees the test cars while learning, so they show how it does on new cars.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)
