import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')


# Sample data (replace with your actual dataset)
data = {
       'sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
       'age': [25, 35, 45, 55, 28, 32, 48, 50, 30, 28],
       'weight': [70, 60, 80, 75, 65, 55, 90, 68, 75, 60],
       'height': [175, 160, 180, 165, 170, 155, 185, 163, 180, 160],  # Add height
       'outcome': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]  # Add outcome
}
df = pd.DataFrame(data)

# Preprocess data (convert categorical features to numerical)
df['sex'] = df['sex'].map({'M': 1, 'F': 0})

# Prepare data
X = df[['sex', 'age', 'weight', 'height']]
y = df['outcome']

# Streamlit app
st.title("ADH Client Outcome Prediction")

st.sidebar.header("Patient Information")
sex = st.sidebar.radio("Sex", ['Male', 'Female'])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=150, value=70)
height = st.sidebar.number_input("Height (cm)", min_value=140, max_value=200, value=170)

if st.button("Predict Outcome"):
    # Convert sex to numerical
    sex_num = 1 if sex == 'Male' else 1  # Corrected the mapping to be consistent with training data

    
    input_data = pd.DataFrame({
        'sex': [sex_num],
        'age': [age],
        'weight': [weight],
        'height': [height]
    })
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("Predicted Outcome: **YES**")
    else:
        st.error("Predicted Outcome: **NO**")
