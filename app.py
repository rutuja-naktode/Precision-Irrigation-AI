import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# AI-Based Precision Irrigation Advisory System
# Beginner-friendly hackathon prototype (software-only)
# ------------------------------------------------------------

st.set_page_config(
    page_title="Precision Irrigation AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 AI-Based Precision Irrigation Advisory System")
st.caption("A software-only prototype that combines farmer inputs, weather forecasts and a demo ML model.")

st.info(
    "Prototype notice: soil moisture is entered manually because this version has no physical sensor. "
    "The ML model is trained on synthetic demonstration data, not field-validated agricultural data. "
    "Use recommendations for demonstration/learning only; confirm irrigation decisions with local agricultural guidance."
)

# Crop settings are approximate demonstration values, not universal agronomic thresholds.
CROP_DATA = {
    "Soybean": {"target": 30.0, "root_depth_m": 0.60, "crop_factor": 1.00},
    "Cotton": {"target": 29.0, "root_depth_m": 0.90, "crop_factor": 1.05},
    "Wheat": {"target": 26.0, "root_depth_m": 0.60, "crop_factor": 0.90},
    "Rice": {"target": 34.0, "root_depth_m": 0.30, "crop_factor": 1.10},
    "Maize": {"target": 30.0, "root_depth_m": 0.80, "crop_factor": 1.00},
    "Chickpea": {"target": 25.0, "root_depth_m": 0.50, "crop_factor": 0.85},
}

STAGE_FACTOR = {
    "Initial": 0.90,
    "Vegetative": 1.00,
    "Flowering / Fruiting": 1.10,
    "Maturity": 0.85,
}

@st.cache_resource
def train_demo_model():
    """
    Trains a Random Forest on synthetic demonstration data.
    Labels are generated from a transparent rule so the app can run
    without downloading a dataset. This is NOT a validated farm model.
    """
    rng = np.random.default_rng(42)
    n = 6000

    moisture = rng.uniform(5, 50, n)
    rain = rng.uniform(0, 50, n)           # forecast rain over next 3 days, mm
    temp = rng.uniform(15, 43, n)          # forecast maximum temperature, C
    target = rng.uniform(20, 38, n)        # crop/stage target moisture, %

    # Synthetic teaching label: lower moisture raises need; rain reduces urgency.
    moisture_gap = target - moisture
    effective_rain = rain * 0.70
    heat_effect = np.maximum(temp - 28, 0) * 0.20
    stress_score = moisture_gap + heat_effect - effective_rain

    labels = np.where(stress_score >= 10, "Irrigate soon",
             np.where(stress_score >= 2, "Monitor", "Wait"))

    X = pd.DataFrame({
        "soil_moisture": moisture,
        "rain_3day_mm": rain,
        "max_temp_c": temp,
        "target_moisture": target
    })
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, labels)
    return model

@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(city_name):
    """Get coordinates from Open-Meteo geocoding, then a 3-day forecast."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_response = requests.get(
        geo_url,
        params={"name": city_name, "count": 1, "language": "en", "format": "json"},
        timeout=10
    )
    geo_response.raise_for_status()
    geo_results = geo_response.json().get("results", [])
    if not geo_results:
        raise ValueError("Location not found. Check the spelling or enter rainfall manually.")

    place = geo_results[0]
    lat, lon = place["latitude"], place["longitude"]

    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_response = requests.get(
        forecast_url,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max",
            "forecast_days": 3,
            "timezone": "auto"
        },
        timeout=10
    )
    forecast_response.raise_for_status()
    daily = forecast_response.json()["daily"]

    weather_df = pd.DataFrame({
        "Date": daily["time"],
        "Rain (mm)": daily["precipitation_sum"],
        "Max temp (°C)": daily["temperature_2m_max"]
    })
    total_rain = float(np.nansum(daily["precipitation_sum"]))
    max_temp = float(np.nanmax(daily["temperature_2m_max"]))
    return place.get("name", city_name), weather_df, total_rain, max_temp

# -------------------- Sidebar: farmer inputs --------------------
st.sidebar.header("🧑‍🌾 Farm details")

location = st.sidebar.text_input("Farm location / nearest city", "Nagpur")
crop = st.sidebar.selectbox("Crop type", list(CROP_DATA.keys()))
soil_type = st.sidebar.selectbox(
    "Soil type",
    ["Black / clayey", "Loamy", "Sandy", "Red soil"]
)
area_acres = st.sidebar.number_input(
    "Farm area (acres)", min_value=0.01, max_value=10000.0,
    value=1.0, step=0.25
)
growth_stage = st.sidebar.selectbox("Crop growth stage", list(STAGE_FACTOR.keys()))
moisture = st.sidebar.slider(
    "Current soil moisture (%)",
    min_value=5, max_value=50, value=24,
    help="Enter a measured or estimated value. Without a sensor, this is manual input."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌦️ Weather input")
use_manual_rain = st.sidebar.checkbox("Enter rainfall manually", value=False)
manual_rain = st.sidebar.number_input(
    "Expected rainfall over next 3 days (mm)",
    min_value=0.0, max_value=300.0, value=0.0, step=1.0
)
manual_temp = st.sidebar.number_input(
    "Expected maximum temperature (°C)",
    min_value=0.0, max_value=55.0, value=32.0, step=1.0
)

# -------------------- Weather retrieval --------------------
weather_ok = False
weather_df = None
weather_error = None
place_name = location
forecast_rain = manual_rain
forecast_temp = manual_temp

if not use_manual_rain:
    try:
        with st.spinner("Fetching 3-day weather forecast..."):
            place_name, weather_df, forecast_rain, forecast_temp = get_weather(location.strip())
        weather_ok = True
    except Exception as exc:
        weather_error = str(exc)
        forecast_rain = manual_rain
        forecast_temp = manual_temp

# -------------------- Calculate recommendation --------------------
crop_info = CROP_DATA[crop]
stage_factor = STAGE_FACTOR[growth_stage]
target_moisture = crop_info["target"] * stage_factor

# Soil-type adjustment is intentionally small and clearly illustrative.
soil_adjustment = {
    "Black / clayey": 1.0,
    "Loamy": 0.0,
    "Sandy": -2.0,
    "Red soil": -1.0
}[soil_type]
target_moisture = float(np.clip(target_moisture + soil_adjustment, 18, 40))

model = train_demo_model()
model_input = pd.DataFrame([{
    "soil_moisture": float(moisture),
    "rain_3day_mm": float(forecast_rain),
    "max_temp_c": float(forecast_temp),
    "target_moisture": float(target_moisture)
}])

prediction = model.predict(model_input)[0]
probabilities = model.predict_proba(model_input)[0]
confidence = float(np.max(probabilities)) * 100

# Approximate water-balance calculation.
# 1 mm of water over 1 m² = 1 litre.
area_m2 = area_acres * 4046.8564224
root_zone_mm = crop_info["root_depth_m"] * 1000
moisture_deficit_mm = max(0.0, (target_moisture - moisture) / 100.0 * root_zone_mm)
effective_rain_mm = forecast_rain * 0.80
net_water_depth_mm = max(0.0, moisture_deficit_mm - effective_rain_mm)

# Assumed application efficiency for a simple demonstration.
application_efficiency = 0.75
gross_water_mm = net_water_depth_mm / application_efficiency
estimated_litres = gross_water_mm * area_m2
estimated_m3 = estimated_litres / 1000

rainwater_litres = effective_rain_mm * area_m2

# -------------------- Main dashboard --------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Crop", crop)
col2.metric("Soil moisture", f"{moisture}%")
col3.metric("3-day rainfall", f"{forecast_rain:.1f} mm")
col4.metric("Forecast max temp.", f"{forecast_temp:.1f} °C")

st.subheader("🌱 Irrigation advisory")

if prediction == "Irrigate soon":
    st.warning("AI result: Irrigate soon — verify soil and field conditions before watering.")
elif prediction == "Monitor":
    st.info("AI result: Monitor — recheck soil moisture and rainfall before deciding.")
else:
    st.success("AI result: Wait — forecast rain / current moisture may be sufficient for now.")

st.write(f"**Model confidence (demo model):** {confidence:.1f}%")
st.caption("This confidence is the model's score on synthetic training data; it is not a real-world accuracy guarantee.")

left, right = st.columns(2)
with left:
    st.markdown("#### Water estimate")
    st.metric("Estimated supplemental water", f"{estimated_litres:,.0f} L")
    st.metric("Equivalent volume", f"{estimated_m3:.2f} m³")
    st.caption(
        "Illustrative estimate based on a simplified root-zone moisture deficit, "
        "80% effective forecast rainfall and 75% application efficiency. "
        "Actual requirement depends on measured field capacity, soil profile, irrigation method and local agronomy."
    )

with right:
    st.markdown("#### Rainfall contribution")
    st.metric("Estimated effective rain contribution", f"{rainwater_litres:,.0f} L")
    st.metric("Crop-stage target moisture (demo)", f"{target_moisture:.1f}%")
    st.caption("Target values are approximate demo settings and should be calibrated for local crop and soil conditions.")

st.markdown("---")
st.subheader("🌦️ Weather forecast")

if weather_ok and weather_df is not None:
    st.success(f"Weather forecast loaded for {place_name}.")
    st.dataframe(weather_df, use_container_width=True, hide_index=True)
elif use_manual_rain:
    st.info("Manual weather values are being used.")
else:
    st.warning(f"Could not load live weather: {weather_error}")
    st.info("You can tick 'Enter rainfall manually' in the sidebar and enter forecast values.")

st.markdown("---")
st.subheader("📊 Farm summary")
summary = pd.DataFrame({
    "Parameter": [
        "Location", "Crop", "Soil type", "Farm area",
        "Growth stage", "Soil moisture", "Target moisture",
        "Forecast rainfall (3 days)", "AI recommendation"
    ],
    "Value": [
        place_name, crop, soil_type, f"{area_acres:.2f} acres",
        growth_stage, f"{moisture}%", f"{target_moisture:.1f}%",
        f"{forecast_rain:.1f} mm", prediction
    ]
})
st.dataframe(summary, use_container_width=True, hide_index=True)

csv = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download farm advisory (CSV)",
    data=csv,
    file_name="irrigation_advisory.csv",
    mime="text/csv"
)

with st.expander("ℹ️ How this prototype works"):
    st.markdown(
        """
        1. Farmer enters crop, growth stage, soil type, farm size and soil moisture.
        2. The app retrieves a 3-day forecast from Open-Meteo, or accepts manual weather values.
        3. A Random Forest classifier gives a demo urgency category: **Irrigate soon**, **Monitor**, or **Wait**.
        4. A separate simplified water-balance calculation estimates supplemental water volume.
        5. The farmer can download a CSV summary.

        **Important:** The model is trained on synthetic data created inside this app so the hackathon demo works offline
        after installation. It is a proof-of-concept, not a field-tested irrigation controller.
        """
    )

st.caption("Built with Python • Streamlit • Scikit-learn • Open-Meteo")

