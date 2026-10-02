// Used only on the static (Netlify) version of the site, where there is no Flask server.
// It does the same calculation as the trained Linear Regression pipeline, in the browser,
// using the numbers exported to model.json by src/build_static_site.py.
// It returns the same shape of answer as Flask's /predict: {price} or {error}.

const modelReady = fetch("/static/model.json").then((response) => response.json());

const LABELS = {
  year: "Year", km_driven: "Km driven", engine_cc: "Engine cc",
  mileage_kmpl: "Mileage kmpl", owners: "Owners",
};

async function predictLocally(car) {
  const model = await modelReady;

  // Same checks as validate() in src/app.py
  const brand = Object.keys(model.car_names).find((b) => model.car_names[b].includes(car.model));
  if (!brand) return { error: "Please choose a car name from the list." };

  for (const column of ["fuel_type", "transmission", "seller_type"]) {
    if (!Object.hasOwn(model.categorical[column], car[column])) {
      return { error: `Please choose a valid ${column.replace("_", " ")}.` };
    }
  }

  const numbers = {};
  for (const [column, [low, high]] of Object.entries(model.limits)) {
    const value = Number(car[column]);
    if (car[column] === "" || Number.isNaN(value)) return { error: `${LABELS[column]} must be a number.` };
    if (value < low || value > high) {
      return { error: `${LABELS[column]} must be between ${low.toLocaleString("en-US")} and ${high.toLocaleString("en-US")}.` };
    }
    numbers[column] = value;
  }

  // log(price) = intercept + (weight x value for every number) + (weight of each chosen category)
  let logPrice = model.intercept;
  for (const [column, weight] of Object.entries(model.numeric)) {
    logPrice += weight * numbers[column];
  }
  const categories = { brand, model: car.model, fuel_type: car.fuel_type,
                       transmission: car.transmission, seller_type: car.seller_type };
  for (const [column, value] of Object.entries(categories)) {
    logPrice += model.categorical[column][value]; // one-hot: only the chosen category's column is 1
  }

  // The model learned log(price), so exp() turns it back into rupees.
  return { price: Math.round(Math.exp(logPrice)) };
}

window.predictLocally = predictLocally;
