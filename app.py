import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="SugarSense",
    page_icon="🩺"
)

@st.cache_resource
def load_artifact():
    return joblib.load("artifacts/sugarsense_model.joblib")

artifact = load_artifact()
model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["features"]

st.title("SugarSense")
st.subheader("Diabetes Risk Screening Demo")

st.warning(
    "Educational demonstration only. "
    "This is not a medical diagnosis."
)

with st.form("screening_form"):
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1
    )
    glucose = st.number_input(
        "Glucose", min_value=0.0, max_value=300.0, value=100.0
    )
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0.0, max_value=200.0, value=70.0
    )
    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0.0, max_value=100.0, value=20.0
    )
    insulin = st.number_input(
        "Insulin", min_value=0.0, max_value=1000.0, value=80.0
    )
    bmi = st.number_input(
        "BMI", min_value=0.0, max_value=80.0, value=25.0
    )
    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.5
    )
    age = st.number_input(
        "Age", min_value=18, max_value=120, value=30
    )

    submitted = st.form_submit_button("Estimate Screening Risk")

if submitted:
    row = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    zero_columns = [
        "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI"
    ]
    row[zero_columns] = row[zero_columns].replace(0, np.nan)

    row["Glucose_BMI"] = row["Glucose"] * row["BMI"]
    row["Glucose_Age"] = row["Glucose"] * row["Age"]
    row["BMI_Age"] = row["BMI"] * row["Age"]

    row = row.reindex(columns=feature_names)

    probability = model.predict_proba(row)[0, 1]
    prediction = probability >= threshold

    st.metric("Estimated model probability", f"{probability:.1%}")

    if prediction:
        st.warning(
            "The model flags this input for further professional assessment."
        )
    else:
        st.info(
            "The model did not flag this input at the selected threshold. "
            "This does not rule out diabetes."
        )