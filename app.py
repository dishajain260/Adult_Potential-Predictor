import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- LOAD TRAINED OBJECTS ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Adult Potential Prediction", layout="centered")
st.title("Adult Potential Prediction")

# ---------------- USER INPUT ----------------
st.header("Enter Your Details:")

# Numeric inputs
age = st.number_input("Age", 18, 100)
fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=0)
education_num = st.number_input("Education Number", 1, 16)
capital_gain = st.number_input("Capital Gain", 0)
capital_loss = st.number_input("Capital Loss", 0)
hours = st.number_input("Hours per week", 1, 100)

# Categorical inputs
education = st.selectbox("Education", label_encoders["education"].classes_)
workclass = st.selectbox("Workclass", label_encoders["workclass"].classes_)
marital = st.selectbox("Marital Status", label_encoders["marital-status"].classes_)
occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)
relationship = st.selectbox("Relationship", label_encoders["relationship"].classes_)
race = st.selectbox("Race", label_encoders["race"].classes_)
country = st.selectbox("Country", label_encoders["country"].classes_)
sex = st.radio("Sex", ["Male", "Female"])

# Salary class input (for salary_sq)
salary_class = st.selectbox("Salary Class", ["<=50K", ">=50K"])
salary_numeric = 1 if salary_class == ">=50K" else 0
salary_sq = salary_numeric ** 2

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    # Encode categorical inputs
    encoded_input = {
        "education": label_encoders["education"].transform([education])[0],
        "workclass": label_encoders["workclass"].transform([workclass])[0],
        "marital-status": label_encoders["marital-status"].transform([marital])[0],
        "occupation": label_encoders["occupation"].transform([occupation])[0],
        "relationship": label_encoders["relationship"].transform([relationship])[0],
        "race": label_encoders["race"].transform([race])[0],
        "country": label_encoders["country"].transform([country])[0],
    }

    # Create DataFrame with derived features
    input_df = pd.DataFrame([{
        "age": age,
        "workclass": encoded_input["workclass"],
        "fnlwgt": fnlwgt,
        "education": encoded_input["education"],
        "education-num_sq": education_num ** 2,
        "marital-status": encoded_input["marital-status"],
        "occupation": encoded_input["occupation"],
        "relationship": encoded_input["relationship"],
        "race": encoded_input["race"],
        "capital-gain_sq": capital_gain ** 2,
        "capital-loss": capital_loss,
        "hours-per-week": hours,
        "country": encoded_input["country"],
        "sex": sex,
        "salary_sq": salary_sq
    }])

    # One-hot encode sex
    sex_clean = sex.strip().title()  # removes spaces, ensures first letter capital
    input_df["sex"] = sex_clean

    sex_encoded = ohe.transform(input_df[["sex"]])
    sex_df = pd.DataFrame(
    sex_encoded,
    columns=ohe.get_feature_names_out(["sex"]),
    index=input_df.index
    )
    input_df = pd.concat([input_df.drop(columns=["sex"]), sex_df], axis=1)


    # ---------------- FIX COLUMN MISMATCH ----------------
    missing_cols = set(scaler.feature_names_in_) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0
    input_df = input_df[scaler.feature_names_in_]

    # ---------------- SCALE ----------------
    input_scaled = scaler.transform(input_df)

    # ---------------- DEBUG INFO ----------------
    # st.subheader("Debug Info")
    # st.write("**Derived Features + Encoded Values:**")
    # st.dataframe(input_df.T)
    # st.write("**Scaled Input:**")
    # st.dataframe(pd.DataFrame(input_scaled, columns=input_df.columns).T)

    # ---------------- PREDICT ----------------
    prediction = model.predict(input_scaled)[0]

    # Show result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("💰 HIGH  Potential")
    else:
        st.warning("📉 LOW  Potential")
