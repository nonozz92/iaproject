import streamlit as st
import requests
import pandas as pd

st.title("Machine Learning API Interface")
st.markdown("""
    ### Documentation
    - [Swagger UI](http://localhost:8000/docs)
    - [ReDoc](http://localhost:8000/redoc)
""")

# Eentraînement du modèle
st.header("Train Model")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:", data.head())

    if st.button("Train Model"):
        features = data.iloc[:, :-1].values.tolist()
        labels = data.iloc[:, -1].tolist()

        response = requests.post("http://localhost:8000/training", json={"features": features, "labels": labels})
        
        if response.status_code == 200:
            st.success("Model trained successfully")
        else:
            st.error("Error training model: " + response.text)

# Prédiction
st.header("Make Prediction")
st.write("Example input format: `6, 148, 72, 35, 0, 33.6, 0.627, 50`")

prediction_data = st.text_area("Enter data for prediction (comma-separated values for each feature)")

if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None

if st.button("Predict"):
    try:
        features = [list(map(float, prediction_data.split(',')))]
        st.write("Sending features to API:", features)
        response = requests.post("http://localhost:8000/predict", json={"features": features})
        
        if response.status_code == 200:
            st.session_state['predictions'] = response.json()["predictions"]
        else:
            st.error("Error making prediction: " + response.text)
    except ValueError as e:
        st.error(f"Please enter valid data: {e}")

# Affichage des prédictions si elles existent
if st.session_state['predictions'] is not None:
    st.write("Prediction:", st.session_state['predictions'])


st.header("Model Info")
response = requests.get("http://localhost:8000/model")

if response.status_code == 200:
    model_info = response.json()
    st.write("Model:", model_info["model"])
    st.write("Library:", model_info["library"])
else:
    st.error("Error fetching model info: " + response.text)
