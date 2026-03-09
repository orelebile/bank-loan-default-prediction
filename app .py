
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model, scaler, and columns
model = joblib.load("loan_risk_logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# --- Fix duplicate and whitespace issues in model columns ---
model_columns = [c.strip() for c in model_columns]        # remove leading/trailing spaces
model_columns = list(dict.fromkeys(model_columns))        # remove duplicates

st.title("Loan Risk Prediction App")
st.write("Enter customer details to predict loan risk")

# --- Numeric inputs ---
numeric_features = ['Income','Age','Experience','Current_Job_Yrs','Current_House_Yrs']
numeric_inputs = {}
for feature in numeric_features:
    numeric_inputs[feature] = st.number_input(feature)

# --- Dynamic dropdowns for categorical features ---
categorical_prefixes = []
for col in model_columns:
    if '_' in col:
        prefix = col.split('_')[0]
        if prefix not in numeric_features and prefix not in categorical_prefixes:
            categorical_prefixes.append(prefix)

categorical_inputs = {}
for prefix in categorical_prefixes:
    # extract all categories for this prefix in model_columns
    options = [c.replace(prefix+'_','') for c in model_columns if c.startswith(prefix+'_')]
    if options:
        categorical_inputs[prefix] = st.selectbox(prefix, options)

# --- Prediction button ---
if st.button("Predict Loan Risk"):

    # Create dataframe from user input
    input_dict = {**numeric_inputs, **categorical_inputs}
    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df)

    # Remove any duplicate columns in input just in case
    input_encoded = input_encoded.loc[:, ~input_encoded.columns.duplicated()]

    # Align input with training columns safely
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    # --- Debug panels ---
    st.subheader("Input Data after One-Hot Encoding & Alignment")
    st.dataframe(input_aligned)

    input_scaled = scaler.transform(input_aligned)
    st.subheader("Scaled Input Features")
    st.dataframe(pd.DataFrame(input_scaled, columns=input_aligned.columns))

    # --- Make prediction ---
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("⚠️ High Loan Risk")
    else:
        st.success("✅ Low Loan Risk")
