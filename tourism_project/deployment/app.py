"""Streamlit app using the exact saved classifier pipeline."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "tourism_package_prediction.joblib"
st.set_page_config(page_title="Tourism Package Prediction")
st.title("Tourism Package Prediction")

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run the training workflow first.")
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

with st.form("prediction_form"):
    contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "King", "Super Deluxe"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.number_input("City Tier", min_value=1, max_value=3, value=1)
    duration = st.number_input("Duration of Pitch", min_value=0.0, value=15.0)
    visitors = st.number_input("Number of People Visiting", min_value=1, value=2)
    followups = st.number_input("Number of Followups", min_value=0, value=2)
    star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
    trips = st.number_input("Number of Trips", min_value=0, value=2)
    passport = st.selectbox("Passport", [0, 1])
    satisfaction = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    car = st.selectbox("Own Car", [0, 1])
    children = st.number_input("Number of Children Visiting", min_value=0, value=0)
    income = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
    submitted = st.form_submit_button("Tourism Prediction")

if submitted:
    input_data = pd.DataFrame([{
        "TypeofContact": contact, "Occupation": occupation, "Gender": gender,
        "ProductPitched": product, "MaritalStatus": marital,
        "Designation": designation, "Age": age, "CityTier": city_tier,
        "DurationOfPitch": duration, "NumberOfPersonVisiting": visitors,
        "NumberOfFollowups": followups, "PreferredPropertyStar": star,
        "NumberOfTrips": trips, "Passport": passport,
        "PitchSatisfactionScore": satisfaction, "OwnCar": car,
        "NumberOfChildrenVisiting": children, "MonthlyIncome": income
    }])
    probability = float(model.predict_proba(input_data)[0, 1])
    prediction = int(probability >= 0.45)
    st.subheader("Prediction Result")
    if prediction:
        st.success("Customer is likely to purchase the tourism package.")
    else:
        st.warning("Customer is unlikely to purchase the tourism package.")
    st.write(f"Estimated purchase probability: {probability:.1%}")
