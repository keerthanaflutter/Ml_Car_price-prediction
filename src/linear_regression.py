"""Build the full model: preprocessing + Linear Regression in one Scikit-learn Pipeline."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor

# Why learn log(price) instead of price?
# A car loses roughly a *percentage* of its value each year (e.g. ~13%), not a fixed rupee amount.
# Rs 1,00,000 is a huge drop for an old hatchback but a small one for a new BMW.
# Taking the logarithm turns "lose 13% per year" into "subtract a constant per year",
# which is exactly the straight-line pattern Linear Regression can learn.
# It also guarantees predictions are never negative, because exp(anything) > 0.
# (Trained on the raw price, the same model scored R² 0.77 and predicted negative prices for old cars.)


def build_model():
    """Return an untrained pipeline: raw car details in, price in rupees out."""
    # TransformedTargetRegressor applies log() to the price before training
    # and exp() to the prediction afterwards, so callers always work in rupees.
    regressor = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp
    )
    # The Pipeline keeps preprocessing and the model together: the saved file
    # cleans and encodes new input exactly the way the training data was handled.
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("linear_regression", regressor),
    ])


def train_model(X_train, y_train):
    """Fit the pipeline: learn one weight per feature column from the training cars."""
    model = build_model()
    model.fit(X_train, y_train)
    return model
