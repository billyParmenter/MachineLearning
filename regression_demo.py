# -*- coding: utf-8 -*-

"""
    FILE NAME:
        regression_demo.py
        
    DESCRIPTION:
        This file demonstrates different regression methods. The regression 
            models are fit to the same data and the R-Squared value is 
            determined. Based on the R2 value the best regression method is
            displayed.
    
    AUTHOR:
        Billy Parmenter
   
    CREATED:
        August 10, 2020
"""


from regression_factory import Regression_factory
from regression_factory import RegressionType



rf = Regression_factory('./Data.csv')

# Dictionary of regression methods for easy epantion
regressionMethods = {
    rf.CreateRegressionMethod(RegressionType.mlr) : "Multiple Linear Regression",
    rf.CreateRegressionMethod(RegressionType.pr) : "Polynomial Regression",
    rf.CreateRegressionMethod(RegressionType.svr) : "Support Vector Regression",
    rf.CreateRegressionMethod(RegressionType.dtr) : "Decision Tree Regression",
    rf.CreateRegressionMethod(RegressionType.rfr) : "Random Forest Regression", 
}



print("Calculating R\u00b2 values for all methods:")

bestR2Score = 0
bestMethod  = ""


# Iterate through the dictonary to find the best method
for regressionMethod in regressionMethods:
    regressionMethod.Regress()
    
    r2Score = regressionMethod.GetR2Score()
    
    if bestR2Score < r2Score:
        bestR2Score = r2Score
        bestMethod  = regressionMethods[regressionMethod]
        
    print("\n\t", regressionMethods[regressionMethod], ": \t", r2Score)    
        
print("\nBest Regression is: ", bestMethod, "\nWith a score of: ", bestR2Score)

k = input("\nPress enter to exit")