from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

def get_baseline_model():
    return DummyClassifier(strategy="most_frequent")

def get_linear_model():
    return LinearRegression()

def get_logistic_regression_model():
    return LogisticRegression(max_iter=200)

def get_random_forest_model_1():
    return RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=7)

def get_random_forest_model_2():
    return RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5)

def get_random_forest_model_3():
    return RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=7)

def get_random_forest_model_4():
    return RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=10)