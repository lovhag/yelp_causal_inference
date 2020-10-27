import pandas as pd
import itertools
from sklearn import preprocessing

class Data:
    def __init__(self, data, treatments, confounders_cont, confounders_cat):
        self.data = data.copy()
        
        # Use dummy encoding
        assert 'stars' in confounders_cat
        assert 'is_positive_sentiment' in confounders_cat
        assert len(confounders_cat) == 2
        stars_dummies = pd.get_dummies(self.data['stars'], drop_first=True, prefix='stars')
        self.data = pd.concat([self.data, stars_dummies], axis=1)
        self.data.drop(columns='stars', inplace=True)
        self.data['is_positive_sentiment'] = self.data['is_positive_sentiment']. astype(int)
        
        self.confounders_cat = stars_dummies.columns.tolist()
        self.confounders_cat.append('is_positive_sentiment')

        self.confounders_cont = confounders_cont
        
        self.treatments = treatments
        self.n_treatments = len(treatments)
        
        self.treatment_groups = list(itertools.product([0, 1], repeat=self.n_treatments))
        
        self._get_data()
        self._get_scaled_data()
        
    def _get_scaled_data(self):
        self.X_train_scaled = dict.fromkeys(self.treatment_groups)
        self.X_test_scaled = dict.fromkeys(self.treatment_groups)
        
        confounders_scaled = preprocessing.scale(self.data[self.confounders_cont])
        confounders_scaled = pd.DataFrame(confounders_scaled, index=self.data.index, columns=self.confounders_cont)
        
        treatment_groups = self.data.groupby(self.treatments)
        
        for tg in self.treatment_groups:            
            self.X_train_scaled[tg] = self.X_train[tg][self.confounders_cat].copy()
            self.X_test_scaled[tg] = self.X_test[tg][self.confounders_cat].copy()
            
            indexes_train = self.X_train[tg].index
            indexes_test = self.X_test[tg].index
            
            self.X_train_scaled[tg][self.confounders_cont] = confounders_scaled.iloc[indexes_train]
            self.X_test_scaled[tg][self.confounders_cont] = confounders_scaled.iloc[indexes_test]
            
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
            
            self.X_train[tg] = temp_data_train[self.confounders_cat].copy()
            self.X_test[tg] = temp_data_test[self.confounders_cat].copy()
            
            self.X_train[tg][self.confounders_cont] = temp_data_train[self.confounders_cont]
            self.X_test[tg][self.confounders_cont] = temp_data_test[self.confounders_cont]
            
            self.Y_train[tg] = temp_data_train['useful']
            self.Y_test[tg] = temp_data_test['useful']
            
            self.Y_train_discrete[tg] = temp_data_train['useful_discrete']
            self.Y_test_discrete[tg] = temp_data_test['useful_discrete']