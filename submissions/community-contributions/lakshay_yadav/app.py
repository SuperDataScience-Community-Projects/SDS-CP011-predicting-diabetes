import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -----------------------------
# ✅ Styling and Page Config
# -----------------------------
st.set_page_config(page_title="🪸 Diabetes Risk Predictor", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #f8f9fa;
        }
        .main, .block-container {
            background-color: #1e1e2f !important;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 0 25px rgba(0, 0, 0, 0.3);
        }
        .stButton>button {
            background-color: #ff6f61;
            color: white;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background-color: #ff3f34;
        }
        h1, h2, h3, .stMarkdown {
            color: #00cec9;
        }
        label, input, .stSelectbox, .stSlider {
            color: #dfe6e9 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🪸 Diabetes Risk Predictor")
st.markdown("Enter the patient information below to predict the likelihood of diabetes.")

# -----------------------------
# ✅ Load model and metadata
# -----------------------------
base_path = os.path.join(os.path.dirname(__file__), "lakshay_data")
model = joblib.load(os.path.join(base_path, "best_model_gradient_boosting.pkl"))
scaler = joblib.load(os.path.join(base_path, "fitted_scaler.pkl"))
encoder = joblib.load(os.path.join(base_path, "fitted_encoder.pkl"))
with open(os.path.join(base_path, "feature_metadata.pkl"), "rb") as f:
    feature_metadata = joblib.load(f)

numeric_features = feature_metadata['numeric_features']
all_features = feature_metadata['feature_names']

# -----------------------------
# 📅 Input Form
# -----------------------------
with st.form("prediction_form"):
    st.subheader("📋 Clinical & Demographic Inputs")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=28.5)
        hbA1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=6.4)
        glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=500, value=155)

    with col2:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        smoking = st.selectbox("Smoking History", ['never', 'former', 'current', 'not current', 'No Info'])
        location = st.text_input("Location", value="California")
        year = st.slider("Year of Record", 2000, 2025, 2020)

    hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
    race = st.selectbox("Race", ['Hispanic', 'Asian', 'AfricanAmerican', 'Caucasian', 'Other'])

    submit = st.form_submit_button("🔍 Predict")

# -----------------------------
# 🔮 Prediction Logic
# -----------------------------
if submit:
    # 1️⃣ Prepare raw input dictionary
    input_dict = {
        'age': age,
        'bmi': bmi,
        'hbA1c_level': hbA1c,
        'blood_glucose_level': glucose,
        'gender': gender,
        'smoking_history': smoking,
        'location': location,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'year': year,
        **{f"race:{r}": 1 if r == race else 0 for r in ['Hispanic', 'Asian', 'AfricanAmerican', 'Caucasian', 'Other']}
    }

    # 2️⃣ Create DataFrame from input
    df_input = pd.DataFrame([input_dict])

    # 3️⃣ Add 'diabetes' column if encoder requires it
    if 'diabetes' not in df_input.columns and 'diabetes' in encoder.feature_names_in_:
        df_input['diabetes'] = 0

    # 4️⃣ Encode features
    try:
        encoded = encoder.transform(df_input)
    except ValueError as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    # 5️⃣ Align encoded output with expected model features
    encoded_columns = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns)

    # Add missing columns if needed
    for col in all_features:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    # Reorder to match model training feature order
    final_df = encoded_df[all_features]


    # ✅ Predict
    prediction = model.predict(final_df)[0]
    probability = model.predict_proba(final_df)[0][1]

    # === Step 6: Display Results
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff6f61"},
            'steps': [
                {'range': [0, 30], 'color': "#44bd32"},
                {'range': [30, 70], 'color': "#fbc531"},
                {'range': [70, 100], 'color': "#e84118"}
            ]
        },
        title={'text': "Diabetes Risk (%)", 'font': {'color': 'white'}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    result_text = "Diabetes" if prediction == 1 else "No Diabetes"
    st.success(f"**Prediction:** {result_text}")
    st.info(f"**Probability:** {probability:.2%}")

    st.markdown("### 📊 Comparison vs Healthy Guidelines")
    healthy_ranges = {
        "BMI": "18.5–24.9",
        "HbA1c Level": "< 5.7%",
        "Blood Glucose Level": "< 140 mg/dL"
    }
    st.write(pd.DataFrame({
        "Parameter": list(healthy_ranges.keys()),
        "Your Value": [bmi, hbA1c, glucose],
        "Healthy Range": list(healthy_ranges.values())
    }))

# -----------------------------
# 📈 Feature Importance
# -----------------------------
st.markdown("### 📈 Feature Importance (Top 5 Features)")

if hasattr(model, "feature_importances_"):
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1][:5]
    top_features = np.array(all_features)[sorted_indices]
    top_importances = feature_importances[sorted_indices]

    fig_feat = go.Figure(go.Bar(
        x=top_importances[::-1],
        y=top_features[::-1],
        orientation='h',
        marker=dict(color='rgba(100,149,237,0.7)', line=dict(color='rgba(58,71,80,1.0)', width=1.5))
    ))

    fig_feat.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        title="Top 5 Important Features",
        height=400,
        margin=dict(l=80, r=20, t=50, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig_feat, use_container_width=True)
else:
    st.warning("The current model does not support feature importance.")