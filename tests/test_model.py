import pytest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.preprocess import load_and_split_data
from src.trainmodel import train_model
from src.evaluate import evaluate_model

@pytest.fixture(scope="module")
def trained_model():
    X_train, X_test, y_train, y_test = load_and_split_data("data/creditcard.csv")
    model = train_model(X_train, y_train)
    return model, X_test, y_test

def test_model_accuracy(trained_model):
    model, X_test, y_test = trained_model
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["accuracy"] > 0.95

def test_model_precision(trained_model):
    model, X_test, y_test = trained_model
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["precision"] > 0.80

def test_model_recall(trained_model):
    model, X_test, y_test = trained_model
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["recall"] > 0.75

def test_model_f1(trained_model):
    model, X_test, y_test = trained_model
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["f1"] > 0.80