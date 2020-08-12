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

from regression_method import Regression_method
from sklearn.ensemble import RandomForestRegressor
 

"""
    CLASS NAME:
        Random_forest_regressor
        
    DESCRIPTION:
        This class is a specialization of the Regression_method class and
            creates a random forest regression model that can be used to 
            generate an R2 value
"""
class Random_forest_regressor(Regression_method):
    
    
    """
        METHOD:
            Train
        DESCRIPTION:
            Used to train the model on the test data            
    """
    def Train(self):
        self.regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        self.regressor.fit(self.X_train, self.y_train)
        
        
        
        
    """
        METHOD:
            Predict
        DESCRIPTION:
            Used to generate the predicted data using the trained regression
                 model
    """    
    def Predict(self):
        self.y_pred = self.regressor.predict(self.X_test)
        np.set_printoptions(precision=2)