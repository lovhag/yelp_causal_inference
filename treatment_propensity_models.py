from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

def get_baseline_model():
    return DummyClassifier(strategy="stratified")

def get_random_forest_model_1():
    return RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=7)

def get_random_forest_model_2():
    return RandomForestClassifier(n_estimators=70, max_depth=None, min_samples_split=7)

def get_random_forest_model_3():
    return RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=5)

def get_random_forest_model_4():
    return RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=7)

def get_random_forest_model_5():
    return RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=7)