# -*- coding: utf-8 -*-

"""
    FILE NAME:
        multiple_linear_regression.py
        
    AUTHOR:
        Billy Parmenter
    
    CREATED:
        August 10, 2020
"""


import numpy as np
import pandas as pd

from regression_method import Regression_method
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


"""
    CLASS NAME:
        Support_vector_regressor
                
    DESCRIPTION:
        This class is a specialization of the Regression_method class and
            creates a support vector regression model that can be used to 
            generate an R2 value

"""
class Support_vector_regressor(Regression_method):

    
    """
        METHOD:
            ImportDataSet
        DESCRIPTION:
            Gets the data set and seperates the dependant and independant 
                columns
        PARAMETERS:
            dataSet - string - the comma seperated values file
        RETURNS:
            X - int[] - The independant values
            y - int[] - The dependant value
    """
    def ImportDataSet(self, dataSet):
        dataset = pd.read_csv(dataSet)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        y = y.reshape(len(y),1)
        return X, y
    
    
    
    
    """
        METHOD:
            Train
        DESCRIPTION:
            Applies feature scaling to the model            
    """
    def FeatureScaling(self):
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.X_train = self.sc_X.fit_transform(self.X_train)
        self.y_train = self.sc_y.fit_transform(self.y_train)
    
    
    
    
    """
        METHOD:
            Train
        DESCRIPTION:
            Used to train the model on the test data            
    """
    def Train(self):    
        self.regressor = SVR(kernel = 'rbf')
        self.regressor.fit(self.X_train, self.y_train.ravel())
        
        
        
        
    """
        METHOD:
            Predict
        DESCRIPTION:
            Used to generate the predicted data using the trained regression
                 model
    """    
    def Predict(self):
        self.y_pred = self.sc_y.inverse_transform(self.regressor.predict(self.sc_X.transform(self.X_test)))
        np.set_printoptions(precision=2)




    """
        METHOD:
            Regress
        DESCRIPTION:
            Applies feature scaling, trains the model and predicts the values          
    """
    def Regress(self):
        self.FeatureScaling()
        self.Train()
        self.Predict()