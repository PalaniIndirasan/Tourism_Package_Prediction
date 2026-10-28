import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_package_prediction.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Please enter the details below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Large Business", "Small Business"])
Designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "King", "Super Deluxe"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Divorced", "Married"])


Age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
CityTier = st.number_input("CityTier", min_value=1, max_value=3, value=1, step=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=127, value=5, step=1)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5, value=1, step=1)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=6, value=1, step=1)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=22, value=1, step=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=1, step=1)
Passport = st.number_input("Passport", min_value=0, max_value=1, value=1, step=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=1, step=1)
OwnCar = st.number_input("OwnCar", min_value=0, max_value=1, value=0, step=1)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=3, value=1, step=1)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, max_value=100000, value=10000, step=1000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'Designation': Designation,
    'MaritalStatus': MaritalStatus,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfTrips': NumberOfTrips,
    'PreferredPropertyStar': PreferredPropertyStar,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}])

# Predict button

if st.button("Tourism Prediction"):

    # Get probability for class 1 (ProdTaken = 1)
    probability = model.predict_proba(input_data)[0][1]

    # Use the same threshold as your training code
    classification_threshold = 0.45

    prediction = int(
        probability >= classification_threshold
    )

    st.subheader("Prediction Result")

    st.write(
        f"Purchase Probability: {probability:.2%}"
    )

    st.write(
        f"Classification Threshold: "
        f"{classification_threshold:.0%}"
    )

    if prediction == 1:
        st.success(
            "Customer is likely to purchase the tourism package."
        )
    else:
        st.warning(
            "Customer is unlikely to purchase the tourism package."
        )
