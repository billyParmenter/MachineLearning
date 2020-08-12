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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


"""
    CLASS NAME:
        Polynomial_regressor
        
    DESCRIPTION:
        This class is a specialization of the Regression_method class and
            creates a polynomial regression model that can be used to 
            generate an R2 value
"""
class Polynomial_regressor(Regression_method):
      
    
    """
        METHOD:
            Train
        DESCRIPTION:
            Used to train the model on the test data            
    """
    def Train(self):
        self.poly_reg = PolynomialFeatures(degree = 4)
        X_poly = self.poly_reg.fit_transform(self.X_train)
        self.regressor = LinearRegression()
        self.regressor.fit(X_poly, self.y_train)




    """
        METHOD:
            Predict
        DESCRIPTION:
            Used to generate the predicted data using the trained regression
                 model
    """
    def Predict(self):
        self.y_pred = self.regressor.predict(self.poly_reg.transform(self.X_test))
        np.set_printoptions(precision=2)