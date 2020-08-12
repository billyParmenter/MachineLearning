# -*- coding: utf-8 -*-

"""
    FILE NAME:
        regression_factory.py
    
    AUTHOR:
        Billy Parmenter
   
    CREATED:
        August 10, 2020
"""


from enum import IntEnum

from multiple_linear_regression import Multiple_linear_regressor
from polynomial_regression import Polynomial_regressor
from support_vector_regression import Support_vector_regressor
from decision_tree_regression import Decision_tree_regressor
from random_forest_regression import Random_forest_regressor


"""
    CLASS NAME:
        RegressionType
        
    DESCRIPTION:
        This class is an int enum used for generating different types of 
            regression models
"""
class RegressionType (IntEnum):
    mlr = 1 # Multiple linear regressor
    pr  = 2 # Polynomial regressor
    svr = 3 # Support vector regressor
    dtr = 4 # Decision tree regressor
    rfr = 5 # Random forest regressor





"""
    CLASS NAME:
        Regression_factory
        
    DESCRIPTION:
        This class is a single location where many types of regression models
            can be instantiated.
"""
class Regression_factory():
    
    
    """
        METHOD:
            __init__
        DESCRIPTION:
            Class constructor, validates and sets the class parameters
        PARAMETERS:
            dataSet - string - the comma seperated values file
            testSize - float - A value between 0 and 1, the percentage of test 
                                values
    """
    def __init__(self, dataSet, testSize = 0.2):
        self.dataSet = dataSet
        self.testSize = testSize
    
    
    
    
    """
        METHOD:
            MLR
        DESCRIPTION:
            Creates and returns a multiple linear regression model object
        RETURN:
            mlr - Multiple_linear_regressor
    """
    def MLR(self):
        self.mlr = Multiple_linear_regressor(self.dataSet, self.testSize)
        return self.mlr
    
    
    
    
    """
        METHOD:
            PR
        DESCRIPTION:
            Creates and returns a polynomial regression model object
        RETURN:
            pr - Polynomial_regressor
    """
    def PR(self):
        self.pr = Polynomial_regressor(self.dataSet, self.testSize)
        return self.pr
    
    
    
    
    """
        METHOD:
            SVR
        DESCRIPTION:
            Creates and returns a support vector regression model object
        RETURN:
            svr - Support_vector_regressor
    """
    def SVR(self):
        self.svr = Support_vector_regressor(self.dataSet, self.testSize)
        return self.svr
    
    
    
    
    """
        METHOD:
            DTR
        DESCRIPTION:
            Creates and returns a decision tree regression model object
        RETURN:
            dtr - Decision_tree_regressor
    """
    def DTR(self):
        self.dtr = Decision_tree_regressor(self.dataSet, self.testSize)
        return self.dtr
    
    
    
    
    """
        METHOD:
            RFR
        DESCRIPTION:
            Creates and returns a random forest regression model object
        RETURN:
            rfr - Random_forest_regressor
    """
    def RFR(self):
        self.rfr = Random_forest_regressor(self.dataSet, self.testSize)
        return self.rfr
    
    
    
    
    # Dictionary for deciding what regressor model to create
    RegressionTypes = {
        1 : MLR, # Multiple linear regressor
        2 : PR,  # Polynomial regressor
        3 : SVR, # Support vector regressor
        4 : DTR, # Decision tree regressor
        5 : RFR, # Random forest regressor
        }
    
    
    
    
    """
        METHOD:
            CreateRegressionMethod
        DESCRIPTION:
            Creates and returns a regression model object bassed on the given
                regression type
        PARAMETERS:
            regressionType - RegressionType -  The type of regression model to 
                                                    create
        RETURN:
            A regression model object
    """
    def CreateRegressionMethod(self, regressionType):
        return self.RegressionTypes[regressionType](self)




    """
        METHOD:
            SetDataSet
        DESCRIPTION:
            Sets the data set file location
        PARAMETERS:
            dataSet - string - the comma seperated values file
    """
    def SetDataSet(self, dataSet):
        self.dataSet = dataSet
        
        
        
        
    """
        METHOD:
            SetTestSize
        DESCRIPTION:
            Checks that the value trying to be set is within range. If it is 
                then the value is set, if not then it prints an error message 
                and uses the last value
        PARAMETERS:
             testSize - float - A value between 0 and 1, the percentage of test 
                                values       
    """
    def SetTestSize(self, testSize):
        if testSize < 1 and testSize > 0:
            self.testSize = testSize
        else:
            print("testSize out of bounds, must be between 0 and 1 exclusively. Using previous value")

    
    
        
        
        