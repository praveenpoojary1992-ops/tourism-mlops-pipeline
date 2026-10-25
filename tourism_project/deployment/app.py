"""Streamlit app: predict whether a customer will purchase the Wellness Tourism Package."""
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_tourism_model_v1.joblib")
THRESHOLD = 0.45  # must match the threshold used in train.py

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🌴")
st.title("🌴 Wellness Tourism Package Predictor")
st.write(
    "Enter the customer and interaction details below to estimate whether the customer "
    "is likely to purchase the **Wellness Tourism Package**."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run the GitHub Actions pipeline to train and commit it.")
    st.stop()

model = load_model()

# ---------------- User inputs ----------------
st.subheader("Customer details")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
with c2:
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=20000.0, step=500.0)
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
    num_children = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=30, value=3)
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

st.subheader("Customer interaction details")
c3, c4 = st.columns(2)
with c3:
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=15.0)
with c4:
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

# ---------------- Save inputs into a dataframe ----------------
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Input data sent to the model"):
    st.dataframe(input_df)

if st.button("Predict"):
    probability = float(model.predict_proba(input_df)[0, 1])
    st.metric("Purchase probability", f"{probability:.1%}")
    if probability >= THRESHOLD:
        st.success("Likely to purchase the Wellness Tourism Package - prioritise this customer.")
    else:
        st.warning("Unlikely to purchase the Wellness Tourism Package.")
