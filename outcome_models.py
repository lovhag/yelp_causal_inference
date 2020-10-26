from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyClassifier

def get_baseline_model():
    return DummyClassifier(strategy="most_frequent")

def get_linear_model():
    return LinearRegression()

def get_logistic_regression_model():
    return LogisticRegression(max_iter=200)