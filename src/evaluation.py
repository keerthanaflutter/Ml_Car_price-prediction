"""Stage 6: measure how close the predicted prices are to the real ones."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred):
    """Return the four standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),  # average size of the error, in rupees
        "MSE": mse,                                  # average squared error (punishes big misses)
        "RMSE": np.sqrt(mse),                        # MSE brought back to rupees
        "R2": r2_score(y_true, y_pred),              # share of price variation explained (1.0 = perfect)
    }


def compare_prices(y_true, y_pred):
    """Put actual and predicted prices side by side for each test car."""
    table = pd.DataFrame({"actual_price": y_true.values, "predicted_price": y_pred.round(-3)}, index=y_true.index)
    table["difference"] = table["predicted_price"] - table["actual_price"]
    table["error_%"] = (table["difference"].abs() / table["actual_price"] * 100).round(1)
    return table
