Arnaud GIBELLI
Hugo Cleret
Faïza Akabli

# Machine Learning API Project

This project provides an API for training and using a machine learning model using FastAPI. It also includes a Streamlit interface for interacting with the API.

## Project Structure

- `api.py`: The file containing the FastAPI application.
- `requirements.txt`: The file listing the dependencies needed for the project.
- `function.py`: The file containing the main functions for preprocessing data, training the model, and making predictions.
- `app.py`: The Streamlit application for interacting with the API.
- `model`: The directory where the trained model is saved.
- `data.csv`: The dataset used for training the model.
- `Notebook.ipynb`: The notebook containing your draft code.

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/nonozz92/iaproject.git
   cd iaproject
   ```

2. **Install the dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

## Running the API

1. **Start the FastAPI server:**

   ```sh
   python -m uvicorn api:app --reload
   ```

   This will start the server at `http://127.0.0.1:8000`.

2. **Access the API documentation:**

   - Open your browser and go to `http://127.0.0.1:8000/docs` for Swagger UI.
   - Alternatively, go to `http://127.0.0.1:8000/redoc` for ReDoc.

## Running the Streamlit App

1. **Start the Streamlit application:**

   ```sh
   streamlit run app.py
   ```

   This will start the Streamlit interface where you can interact with the API. Open your browser and go to the URL provided by the Streamlit command (typically `http://localhost:8501`).

## Using the API

### Training the Model

1. **Upload a CSV file** with your dataset via the Streamlit interface.
2. **Click the "Train Model" button** to send the data to the API and train the model.

### Making Predictions

1. **Enter the data** for prediction in the text area provided in the Streamlit interface.
2. **Click the "Predict" button** to get the prediction from the trained model.
