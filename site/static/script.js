// Sends the form to Flask (POST /predict) and shows the price it returns.
const form = document.getElementById("car-form");
const button = form.querySelector("button");
const resultCard = document.getElementById("result");
const emptyView = document.getElementById("result-empty");
const successView = document.getElementById("result-success");
const errorView = document.getElementById("result-error");
const priceText = document.getElementById("price");
const errorText = document.getElementById("error-message");
const carText = document.getElementById("result-car");

// Show exactly one of the three result views: "empty", "success" or "error".
function showResult(view) {
  emptyView.hidden = view !== "empty";
  successView.hidden = view !== "success";
  errorView.hidden = view !== "error";
  resultCard.classList.toggle("success", view === "success");
  // On phones the result card sits below the form, so scroll it into view.
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault(); // stop the browser from reloading the page

  // Turn the form fields into an object like { model: "Creta", year: "2019", ... }
  const carDetails = Object.fromEntries(new FormData(form));

  button.disabled = true;
  button.textContent = "Predicting…";
  try {
    let data;
    if (window.predictLocally) {
      // Static version (Netlify): no server, so predict in the browser (static/predict_local.js).
      data = await window.predictLocally(carDetails);
    } else {
      // Flask version: send the details to the server and get the prediction back.
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(carDetails),
      });
      data = await response.json();
    }

    if (data.price !== undefined) {
      // Indian number format: 924516 -> ₹9,24,516
      priceText.textContent = "₹" + data.price.toLocaleString("en-IN");
      // Show which car was priced, e.g. "2019 Hyundai Creta · 45,000 km"
      const carName = form.elements.model.selectedOptions[0].text;
      const km = Number(carDetails.km_driven).toLocaleString("en-IN");
      carText.textContent = `${carDetails.year} ${carName} · ${km} km`;
      showResult("success");
    } else {
      errorText.textContent = data.error;
      showResult("error");
    }
  } catch {
    errorText.textContent = "Could not get a prediction. Please check your connection and try again.";
    showResult("error");
  }
  button.disabled = false;
  button.textContent = "Predict Used Car Price";
});
