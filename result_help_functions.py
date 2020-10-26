import pandas as pd
import itertools
from sklearn import preprocessing

class Data:
    def __init__(self, data, treatments, confounders, include_stars=True):
        self.data = data
        self.treatments = treatments
        self.n_treatments = len(treatments)
        self.treatment_groups = list(itertools.product([0, 1], repeat=self.n_treatments))
        self.confounders = confounders
        self.include_stars = include_stars
        self._get_data()
        self._get_scaled_data()
        
    def _get_scaled_data(self):
        self.X_train_scaled = dict.fromkeys(self.treatment_groups)
        self.X_test_scaled = dict.fromkeys(self.treatment_groups)
        
        confounders_scaled = preprocessing.scale(self.data[self.confounders])
        confounders_scaled = pd.DataFrame(confounders_scaled, index=self.data.index, columns=self.confounders)
        
        treatment_groups = self.data.groupby(self.treatments)
        
        for tg, indexes in treatment_groups.groups.items():
            filter_col = [col for col in self.X_train[tg] if col.startswith('stars')]
            
            self.X_train_scaled[tg] = self.X_train[tg][filter_col].copy()
            self.X_test_scaled[tg] = self.X_test[tg][filter_col].copy()
            
            indexes_train = self.X_train[tg].index
            indexes_test = self.X_test[tg].index
        
            self.X_train_scaled[tg][self.confounders] = confounders_scaled.iloc[indexes_train]
            self.X_test_scaled[tg][self.confounders] = confounders_scaled.iloc[indexes_test]
            
    def _get_data(self):
        self.X_train = dict.fromkeys(self.treatment_groups)
        self.X_test = dict.fromkeys(self.treatment_groups)
        
        self.Y_train = dict.fromkeys(self.treatment_groups)
        self.Y_test = dict.fromkeys(self.treatment_groups)
        
        self.Y_train_discrete = dict.fromkeys(self.treatment_groups)
        self.Y_test_discrete = dict.fromkeys(self.treatment_groups)
        
        treatment_groups = self.data.groupby(self.treatments)
        
        for tg, indexes in treatment_groups.groups.items():
            temp_data_train = self.data.iloc[indexes].loc[self.data['test'] == 0]
            temp_data_test = self.data.iloc[indexes].loc[self.data['test'] == 1]
            
            if self.include_stars:
                self.X_train[tg] = pd.get_dummies(temp_data_train['stars'], drop_first=True, prefix='stars')
                self.X_test[tg] = pd.get_dummies(temp_data_test['stars'], drop_first=True, prefix='stars')
            
            self.X_train[tg][self.confounders] = temp_data_train[self.confounders]
            self.X_test[tg][self.confounders] = temp_data_test[self.confounders]
            
            self.Y_train[tg] = temp_data_train['useful']
            self.Y_test[tg] = temp_data_test['useful']
            
            self.Y_train_discrete[tg] = temp_data_train['useful_discrete']
            self.Y_test_discrete[tg] = temp_data_test['useful_discrete']