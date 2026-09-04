import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model.joblib")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️")
st.title("Tourism Package Purchase Prediction")
st.write("Enter customer details to estimate the likelihood of purchasing the wellness tourism package.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=2)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0, step=1.0)
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Executive", "Free Lancer"],
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.number_input(
        "Number of People Visiting", min_value=1, value=2, step=1
    )
    number_of_followups = st.number_input(
        "Number of Follow-ups", min_value=0.0, value=3.0, step=1.0
    )
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0], index=1)
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    number_of_trips = st.number_input("Number of Trips", min_value=0.0, value=2.0, step=1.0)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda value: "Yes" if value else "No")
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda value: "Yes" if value else "No")
    number_of_children_visiting = st.number_input(
        "Number of Children Visiting", min_value=0.0, value=1.0, step=1.0
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0, step=1000.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": type_of_contact,
                "CityTier": city_tier,
                "DurationOfPitch": duration_of_pitch,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": number_of_person_visiting,
                "NumberOfFollowups": number_of_followups,
                "ProductPitched": product_pitched,
                "PreferredPropertyStar": preferred_property_star,
                "MaritalStatus": marital_status,
                "NumberOfTrips": number_of_trips,
                "Passport": passport,
                "PitchSatisfactionScore": pitch_satisfaction_score,
                "OwnCar": own_car,
                "NumberOfChildrenVisiting": number_of_children_visiting,
                "Designation": designation,
                "MonthlyIncome": monthly_income,
            }
        ]
    )

    purchase_probability = model.predict_proba(input_data)[0, 1]
    prediction = int(purchase_probability >= 0.5)

    if prediction:
        st.success(f"Likely to purchase the package ({purchase_probability:.1%} probability).")
    else:
        st.info(f"Unlikely to purchase the package ({purchase_probability:.1%} probability).")
