import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def preprocess_data(features, labels=None):
    """
    Function to preprocess input data for training or prediction.
    :param features: List of feature lists.
    :param labels: List of labels (optional, only for training).
    :return: DataFrame for features, Series for labels (if provided).
    """
    X = pd.DataFrame(features)
    y = pd.Series(labels) if labels is not None else None
    return X, y

def train_model(features, labels, model_path="model/model.joblib"):
    """
    Function to train a logistic regression model and save it to a file.
    :param features: DataFrame of features.
    :param labels: Series of labels.
    :param model_path: Path to save the trained model.
    :return: Trained model.
    """
    model = LogisticRegression(max_iter=200)
    model.fit(features, labels)
    joblib.dump(model, model_path)
    return model

def load_model(model_path="model/model.joblib"):
    """
    Function to load a trained model from a file.
    :param model_path: Path to the saved model.
    :return: Loaded model.
    """
    model = joblib.load(model_path)
    return model

def predict(model, features):
    """
    Function to make predictions using a trained model.
    :param model: Trained model.
    :param features: DataFrame of features.
    :return: List of predictions.
    """
    predictions = model.predict(features)
    return predictions.tolist()
