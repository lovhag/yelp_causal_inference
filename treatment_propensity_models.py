from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

def get_baseline_model():
    return DummyClassifier(strategy="stratified")

def get_random_forest_model_1():
    return RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=7)