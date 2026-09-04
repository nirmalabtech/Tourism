import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model.joblib")

st.set_page_config(page_title="Tourism Package Prediction", page_icon=":airplane:", layout="centered")
st.title("Tourism Package Purchase Prediction")
st.write(
    "Enter customer details to estimate the likelihood of purchasing the "
    "Wellness Tourism Package."
)


@st.cache_resource
def load_model():
    """Load the model once per session, with a friendly error if it is missing."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Trained model not found. Run the GitHub Actions pipeline so it "
            "commits `best_tourism_model.joblib` into this folder, then reload."
        )
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        st.error(f"The model file could not be loaded: {error}")
        st.stop()


model = load_model()

with st.form("prediction_form"):
    st.subheader("Customer details")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0.0, value=23000.0, step=1000.0
    )

    st.subheader("Trip preferences")
    number_of_person_visiting = st.number_input(
        "Number of People Visiting", min_value=1, max_value=10, value=3, step=1
    )
    number_of_children_visiting = st.number_input(
        "Number of Children Visiting (under 5)", min_value=0.0, max_value=10.0,
        value=1.0, step=1.0
    )
    number_of_trips = st.number_input(
        "Number of Trips per Year", min_value=0.0, max_value=50.0, value=3.0, step=1.0
    )
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0], index=0)
    passport = st.selectbox(
        "Has Passport?", [0, 1], format_func=lambda value: "Yes" if value else "No"
    )
    own_car = st.selectbox(
        "Owns a Car?", [0, 1], format_func=lambda value: "Yes" if value else "No"
    )

    st.subheader("Sales interaction")
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=0.0, max_value=180.0, value=13.0, step=1.0
    )
    number_of_followups = st.number_input(
        "Number of Follow-ups", min_value=0.0, max_value=20.0, value=4.0, step=1.0
    )
    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score", min_value=1, max_value=5, value=3
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    # Collect the inputs into a single-row dataframe. The column names must match
    # the names used during training; the ColumnTransformer selects by name, so
    # ordering does not matter, but spelling does.
    input_data = pd.DataFrame(
        [
            {
                "Age": float(age),
                "TypeofContact": type_of_contact,
                "CityTier": int(city_tier),
                "DurationOfPitch": float(duration_of_pitch),
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": int(number_of_person_visiting),
                "NumberOfFollowups": float(number_of_followups),
                "ProductPitched": product_pitched,
                "PreferredPropertyStar": float(preferred_property_star),
                "MaritalStatus": marital_status,
                "NumberOfTrips": float(number_of_trips),
                "Passport": int(passport),
                "PitchSatisfactionScore": int(pitch_satisfaction_score),
                "OwnCar": int(own_car),
                "NumberOfChildrenVisiting": float(number_of_children_visiting),
                "Designation": designation,
                "MonthlyIncome": float(monthly_income),
            }
        ]
    )

    st.subheader("Model input")
    st.dataframe(input_data, use_container_width=True)

    purchase_probability = float(model.predict_proba(input_data)[0, 1])
    prediction = int(purchase_probability >= 0.5)

    st.subheader("Prediction")
    st.metric("Purchase probability", f"{purchase_probability:.1%}")
    st.progress(min(max(purchase_probability, 0.0), 1.0))

    if prediction:
        st.success(
            "Likely to purchase the Wellness Tourism Package. "
            "Recommended for the outbound campaign."
        )
    else:
        st.info(
            "Unlikely to purchase the Wellness Tourism Package. "
            "Deprioritise for this campaign."
        )
