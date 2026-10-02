# Used Car Price Predictor

A beginner-friendly machine learning project that predicts the **selling price of a used car** from its details (brand, model, year, kilometers driven, fuel type and more), with a small website where you can try it.

The model is **Linear Regression** (Scikit-learn), trained on 400 example car sales. The website uses **Flask** (Python) on the back end and plain **HTML, CSS and JavaScript** on the front end.

## Project structure

```
used_car_price_predictor/
├── data/used_cars.csv          Dataset: 400 used-car sales
├── models/car_price_model.pkl  The trained model (created by train_model.py)
├── src/
│   ├── preprocessing.py        Load + inspect data, missing-value filling, one-hot encoding, train/test split
│   ├── linear_regression.py    Builds the Pipeline (preprocessing + Linear Regression) and trains it
│   ├── evaluation.py           MAE, MSE, RMSE, R² and the actual-vs-predicted table
│   ├── train_model.py          Runs the whole ML workflow and saves the model
│   └── app.py                  Flask web server: the page + POST /predict
├── templates/index.html        The web page
├── static/style.css            Page styling
├── static/script.js            Sends the form to Flask and shows the price
└── requirements.txt            Python packages needed
```

## How to run it

Run every command from the project folder (`used_car_price_predictor/`).

### 1. Activate the virtual environment

A virtual environment keeps this project's packages separate from the rest of your computer.

This project's environment lives one folder up, at `ML_Learning/ml_env`:

```bash
source ../ml_env/bin/activate
```

On Windows: `..\ml_env\Scripts\activate`

If you don't have one yet, create it first with `python3 -m venv ../ml_env`, then activate it.
When it is active, your terminal prompt starts with `(ml_env)`.

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

This installs pandas, numpy, scikit-learn, flask and joblib.

### 3. Train the model

```bash
python src/train_model.py
```

This prints every step of the workflow (inspection, training, evaluation) and saves the trained model to `models/car_price_model.pkl`. Run it again whenever you change the data or the code in `src/`.

### 4. Start the Flask website

```bash
python src/app.py
```

Leave this terminal open; the website runs as long as this command runs. Press `Ctrl+C` to stop it.

### 5. Open the website

Go to **http://127.0.0.1:5000** in your browser, fill in the car details and click **Predict Price**.

> **Mac users:** if the page is blank or shows "403 Forbidden", the AirPlay Receiver is using port 5000. Either turn it off (System Settings → General → AirDrop & Handoff → AirPlay Receiver), or start Flask on another port with `PORT=5001 python src/app.py` and open http://127.0.0.1:5001.

## How the prediction works

```
Browser form ──(JSON)──▶ Flask POST /predict ──▶ validate ──▶ saved Pipeline ──▶ price ──(JSON)──▶ Browser shows ₹ price
```

1. **You fill in the form.** The dropdown options (car names, fuel types, …) come from the categories the model saw during training, so you can only choose values it knows.
2. **JavaScript sends the details to Flask.** `static/script.js` stops the page from reloading and sends the form as JSON to `POST /predict`, for example:
   ```json
   {"model": "Creta", "year": "2019", "km_driven": "45000", "fuel_type": "Diesel",
    "seller_type": "Dealer", "transmission": "Manual", "owners": "1",
    "mileage_kmpl": "21.4", "engine_cc": "1493"}
   ```
3. **Flask validates the input** (`validate()` in `src/app.py`): the car name and categories must be known, and every number must be a number within a sensible range (e.g. year 1990 to this year). The brand is looked up from the car name. If something is wrong, Flask replies `{"error": "..."}` with status 400 and the page shows the message.
4. **The saved model predicts.** Flask loads `models/car_price_model.pkl` once at startup. The file holds the whole Pipeline, so the new car goes through exactly the same preprocessing as the training data, then Linear Regression, and comes out as a price in rupees.
5. **Flask returns JSON**, e.g. `{"price": 611578}`, and the page shows **Estimated Used Car Price: ₹ 6,11,578**.

## The dataset

Each row is one used car and the price it sold for. The data was generated for learning: the car models, engine sizes, mileages and price levels are based on real Indian-market cars, and prices follow realistic depreciation, but the rows are not actual sales records. It covers 11 brands (Maruti Suzuki, Hyundai, Honda, Toyota, Tata, Mahindra, Kia, Ford, Volkswagen, BMW, Mercedes-Benz), 23 models, and manufacturing years 2010–2024. A few values are deliberately missing, as in real-world data, so the preprocessing has something to fix.

| Column | Meaning | Example | Website field |
|---|---|---|---|
| `brand` | Manufacturer | Hyundai | Car Name (brand is looked up from the model) |
| `model` | Car model | Creta | Car Name |
| `year` | Manufacturing year | 2019 | Year |
| `km_driven` | Total kilometers on the odometer | 45000 | Kilometers Driven |
| `fuel_type` | Petrol, Diesel or CNG | Diesel | Fuel Type |
| `transmission` | Manual or Automatic | Manual | Transmission |
| `engine_cc` | Engine size in cc | 1493 | Engine |
| `mileage_kmpl` | Fuel efficiency in km per litre | 21.4 | Mileage |
| `owners` | Number of previous owners | 1 | Owner |
| `seller_type` | Dealer or Individual | Dealer | Seller Type |
| **`selling_price`** | **Target: the price in rupees** | 611000 | (predicted) |

The dataset has no max power or seats columns, so the website does not ask for them.

## The ML workflow (`src/train_model.py`)

1. **Load** `used_cars.csv` into a pandas DataFrame.
2. **Inspect** it: size, first rows, column types, missing values, number ranges.
3. **Clean:** drop rows without a price (there are none, but a row without a price can't teach the model anything).
4. **Split** into 80% training cars (320) and 20% test cars (80). Test cars are never shown to the model while it learns, so they show how it does on new cars.
5. **Train** a Scikit-learn **Pipeline** with two steps:
   - **Preprocessing** (`ColumnTransformer`):
     - Number columns: missing values → the column's *median* (not pulled up by a few luxury cars).
     - Category columns: missing values → the most common value, then **OneHotEncoder**. Each category becomes its own 0/1 column: `fuel_type = Diesel` → `fuel_type_Diesel = 1`, `fuel_type_Petrol = 0`, `fuel_type_CNG = 0`.
   - **Linear Regression**, which learns one weight per column (46 in total).
6. **Predict** prices for the 80 test cars.
7. **Evaluate** (next section).
8. **Save** the whole pipeline with `joblib` to `models/car_price_model.pkl`.

Keeping preprocessing and the model in one Pipeline means the website never has to repeat the preprocessing by hand. One saved file does everything.

**Why the model learns log(price):** a car loses roughly a *percentage* of its value every year, not a fixed amount. Losing Rs 1 lakh is huge for a Rs 2 lakh hatchback and small for a Rs 40 lakh BMW. The logarithm turns "lose 13% per year" into "subtract a fixed amount per year", which is the straight-line pattern Linear Regression can learn. `TransformedTargetRegressor` applies `log()` before training and `exp()` after predicting, so predictions come out in rupees and can never be negative.

## How the model is evaluated

| Metric | What it means | Result |
|---|---|---|
| **MAE** (Mean Absolute Error) | Average size of the mistake, in rupees | Rs 46,224 |
| **MSE** (Mean Squared Error) | Average of squared mistakes; big misses count much more. Unit is "rupees squared", so it's hard to read on its own | 5,576,751,763 |
| **RMSE** (Root Mean Squared Error) | Square root of MSE, back in rupees | Rs 74,678 |
| **R²** (R-squared) | Share of the price variation the model explains. 1.0 = perfect, 0 = no better than guessing the average | 0.986 |

Lower MAE/MSE/RMSE is better; higher R² is better. Errors of a few percent are expected: two cars with identical details can still sell for different prices because of condition, colour or how keen the seller is, which the dataset doesn't record.

## Ideas to try next

- Plot actual vs predicted prices with matplotlib.
- Print the learned weights to see which features raise or lower the price most.
- Compare against a Decision Tree or Random Forest regressor.
