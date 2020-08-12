# -*- coding: utf-8 -*-

"""
    FILE NAME:
        multiple_linear_regression.py
        
    AUTHOR:
        Billy Parmenter
    
    CREATED:
        August 10, 2020
"""


import pandas as pd


"""
    CLASS NAME:
        Regression_method
        
    DESCRIPTION:
        This class is the base class for the regression models. Some methods 
            are required but the logic can not be shared so the base class rises
            a not implemented exception
"""
class Regression_method():
    
    
    dataSet = ''
    testSize = 0.2
    
    
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
        return X, y



        
    """
        METHOD:
            SplitDataSet
        DESCRIPTION:
            Splits the data into two sets, train and test. The size of each is 
                based on the testSize value
        PARAMETERS:
            X - int[] - The independant values
            y - int[] - The dependant value
            testSize - float - A value between 0 and 1, the percentage of test 
                                values
        RETURNS:
            X_train - int[] - Independant values to train the model with
            X_test  - int[] - Independant values to test the model with
            y_train - int[] - Dependant values to train the model with
            y_test  - int[] - Dependant values to test the model with
    """
    def SplitDataSet(self, X, y, testSize):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)
        return X_train, X_test, y_train, y_test




    """
        METHOD:
            GetR2Score
        DESCRIPTION:
            determines the R-Squared value of the model based on the predicted 
                y values and the actual y values
        PARAMETERS:
            y_pred - int[] - Dependant values that the model generated
            y_test - int[] - Dependant values the model was tested with
        RETURNS:
            float - the R-Squared value, the closer to 1 the better the model
    """
    def GetR2Score(self):
        from sklearn.metrics import r2_score
        return (r2_score(self.y_test, self.y_pred))
    
    
    
    
    """
        METHOD:
            Init
        DESCRIPTION:
            Initializes the model by importing the data set and splitting it 
                into test and train sets
        PARAMETERS:
            dataSet - string - the comma seperated values file
            testSize - float - A value between 0 and 1, the percentage of test 
                                values
        RETURNS:
            X_train - int[] - Independant values to train the model with
            X_test  - int[] - Independant values to test the model with
            y_train - int[] - Dependant values to train the model with
            y_test  - int[] - Dependant values to test the model with
    """
    def Init(self, dataSet, testSize):
        X, y = self.ImportDataSet(dataSet)
        return self.SplitDataSet(X, y, testSize)
    
    
    
    
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
    def __init__(self, dataSet, testSize):
        self.SetTestSize(testSize)
        self.SetDataSet(dataSet)
    
    
    
    
    """
        METHOD:
            Train
        DESCRIPTION:
            This method needs to be implemented in the child regression class.
                Used to train the model on the test data            
    """
    def Train(self):
        raise NotImplementedError( "Train is not implemented." )
    
    
    
    
    """
        METHOD:
            Predict
        DESCRIPTION:
            This method needs to be implemented in the child regression class.
                 Used to generate the predicted data using the trained regression
                 model
    """
    def Predict(self):
        raise NotImplementedError( "Predict is not implemented." )
    
    
    
    
    """
        METHOD:
            Plot
        DESCRIPTION:
            This method needs to be implemented in the child regression class.
                Used to generate a plot of the predicted and actual values            
    """
    def Plot(self):
        raise NotImplementedError( "Plot is not implemented." )

        
        
        
    """
        METHOD:
            Regress
        DESCRIPTION:
            Trains the model and predicts the values          
    """
    def Regress(self):
        self.Train()
        self.Predict()

        
        
        
    """
        METHOD:
            LoadDataSet
        DESCRIPTION:
            This method needs to be implemented in the child regression class.
                Ges X_train, X_test, y_train, y_test values
    """
    def LoadDataSet(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.Init(self.dataSet, self.testSize)

        
        
        
    """
        METHOD:
            SetDataSet
        DESCRIPTION:
            Sets the data set file location and loads it. Need to add check for loading the file
        PARAMETERS:
            dataSet - string - the comma seperated values file
    """
    def SetDataSet(self, dataSet):
        self.dataSet = dataSet
        self.LoadDataSet()
        
        
        
        
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
