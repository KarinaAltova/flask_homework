#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import pickle


# In[2]:


DF_URL = 'https://raw.githubusercontent.com/devrepublik/data-science-course/master/data/regression/InsuranceCost.csv'
df = pd.read_csv(DF_URL)
df.head()


# In[3]:


class DataPreprocessor(object):
    """
    This is a class for your own DataPreprocessor. 
    It gathers many methods of data preprocessing into one class.
    You may add any other features you want.
    """
  
    def __init__(self):
        """
          Initialize all needed params here.
        """

        pass
    
    def scale(self, X, method='min_max'):
        """
          Scale function. Performs scaling with the specified method. 
          Returns scaled result.

          Args:
            X: array-like input values.
            method: {min_max, max_abs, standart} default min_max. 
              Defined which method of scaling to use.
          Return:
            scaled: scaled data.
        """
        assert method in ['min_max', 'max_abs', 'standard'], 'Check your method'

        if (method == 'min_max'):
            df1 = (X - X.min())/(X.max() - X.min())
            
        elif (method == 'max_abs'):
            df1 = X/(X.abs().max())
            
        elif (method == 'standart'):
            df1 = (X - X.mean())/(X.std())
            
        else:
            return 'Sorry, the method is not correct :('
        

        return df1
    
        

    
    def normalize(self, X, method='max_norm'):
        """
          Normalize function. Performs normalization with the specified method. 
          Returns normalized result.

          Args:
            X: array-like input values.
            method: {max_norm, l1, l2} default max_norm. 
              Defined which method of normalization to use.
          Return:
            normalized: normalized data.
        """
        assert method in ['max_norm', 'l1', 'l2'], 'Check your method !!!'
        
        if method == 'max_norm':
            return X/(X.max())
        
        if method == 'l1':
            return X/(X.abs().values.sum())
        
        else:
            return X/(np.sqrt(X.pow(2).sum()))
        

    
    def encode_onehot(self, X):
        """
      Encoding function. Performs One-hot encoding on given data. 
      Returns encoded result.

      Args:
        X: input values.
      Return:
        encoded: encoded data.
        """
        uniq_item = X.unique()


        encode_df = pd.DataFrame()

        for item in uniq_item:
            encode_df[X.name + '_' + str(item)] = [1 if i==item else 0 for i in X]

        return encode_df

        
    
    def encode_labels(self, X, mapping=None):
        """
          Encoding function. Performs Label encoding on given data. 
          If the mapping is specified, applies it.
          Works on multiple features, even when the mapping is given. 
          Returns encoded result.

          Args:
            X: input values.
          Return:
            encoded: encoded data.
        """
        if mapping == None:
            uniq_item = X.unique()
            uniq_item.sort()
            mapping = {item:i for (i, item) in enumerate(uniq_item)}
        encode_df = pd.DataFrame()
        encode_df[X.name + '_labels'] = X.apply(lambda val: mapping[val])    

        return encode_df
    
    def train_test_split(self, X, y, test_size=0.2, shuffle=True, random_state=42):
        """
      Performs spliting of data

      Args:
        X: samples without labels.
        y: labels.
        test_size: represent the proportion of the dataset to include in 
          the test split. If float, should be between 0.0 and 1.0 and represent 
          the proportion of the dataset to include in the test split. 
          If int, represents the absolute number of test samples. 
        shuffle: whether or not to shuffle the data before splitting.
        random_state: random seed.
      Return:
        X_train: train data.
        X_test: test data.
        y_train: train labels.
        y_test: test labels.
        
        """
        assert shuffle in [True, False], 'shuffle is incorrect'
        assert 0<= test_size < len(X), 'test_size is incorrect'
        
        full_set = pd.concat([X, y], axis=1)
        
        if shuffle == True:
            full_set = full_set.sample(frac=1).reset_index(drop=True)
        
        if test_size <=1:
            test_size = len(full_set) * test_size
            
        test_size = int(test_size)
            
        X_train = full_set.iloc[test_size:, :-1]
        X_test = full_set.iloc[:test_size, :-1] 
        y_train = full_set.iloc[test_size:, -1] 
        y_test = full_set.iloc[:test_size, -1] 
        
        return X_train, X_test, y_train, y_test
            


# In[4]:


model_data = DataPreprocessor()
df = pd.concat([df, model_data.encode_labels(df['sex'])], axis=1)
del df['sex']
df = pd.concat([df, model_data.encode_labels(df['smoker'])], axis=1)
del df['smoker']
df = pd.concat([df, model_data.encode_onehot(df['region'])], axis=1)
del df['region']


# In[5]:


df['age'] = pd.qcut(df['age'], 8, labels=[0, 1, 2, 3, 4,5,6,7])


# In[6]:


X_train, X_test, y_train, y_test = model_data.train_test_split(df.iloc[:, df.columns != 'charges'] , df['charges'], test_size = 10)


# In[7]:


X_train.head()


# In[11]:


model = tree.DecisionTreeRegressor(criterion='mse')
model.fit(X_train, y_train)


# In[12]:


# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))






