from sklearn.linear_model import LinearRegression, LogisticRegression

def get_linear_model():
    return LinearRegression()

def get_logistic_regression_model():
    return LogisticRegression(max_iter=200)
