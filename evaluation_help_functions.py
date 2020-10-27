import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from result_help_functions import Data

def get_pandas_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna(axis=0)
    df.reset_index(inplace=True, drop=True)
    
    print("Data loaded! Loaded data info:")
    df.info()
    return df

def get_evaluation_data(filename, treatments, confounders):
    df = get_pandas_data(filename)

    return Data(df, treatments, confounders)


def get_treatment_model_crossval_scores(models, treatment_groups, X_data, Y_data):
    score_dict = {}

    for treatment in treatment_groups:
        score_dict[treatment] = get_treatment_model_crossval_scores_basic(models, X_data[treatment], Y_data[treatment])
    return score_dict

def get_treatment_model_crossval_scores_basic(models, X_data, Y_data):
    score_dict = {}

    for model_name, model in models.items():
        try: 
            score_dict[model_name] = cross_val_score(model, X_data, Y_data, cv=5)
        except Exception as e:
            print(f"Model evaluation failed for model: {model_name}")
            print("Caught exception: ")
            print(e)
    return score_dict

def get_treatment_model_crossval_predictions_basic(models, X_data, Y_data):
    predict_dict = {}

    for model_name, model in models.items():
        try: 
            predict_dict[model_name] = cross_val_predict(model, X_data, Y_data, cv=5)
        except Exception as e:
            print(f"Model evaluation failed for model: {model_name}")
            print("Caught exception: ")
            print(e)
    return predict_dict

def get_mean_score_per_model(score_dict):
    mean_score = {}
    for treatment, model_scores in score_dict.items():
        for model_name, model_score in model_scores.items():
            if model_name in mean_score:
                mean_score[model_name] = (mean_score[model_name]+np.mean(model_score))/2
            else:
                mean_score[model_name] = np.mean(model_score)
    return mean_score

def plot_treatment_model_scores(score_dict):
    for treatment, model_scores in score_dict.items():
        plot_model_scores(model_scores, treatment)

def plot_model_scores(model_scores, plot_title):
    try:
        plt.figure(figsize=(15,5))
        plt.boxplot(x=model_scores.values(), labels=model_scores.keys())
        plt.title(plot_title)
        plt.show()
    except Exception as e:
        print(f"Score plot failed for plot title: {plot_title}")
        print("Caught exception: ")
        print(e)