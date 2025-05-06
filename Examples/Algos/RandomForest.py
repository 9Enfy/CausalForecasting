import numpy as np
from matplotlib import pyplot as plt
import sklearn
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def RandomForest(dataframe, features, target):
    r"""
    Preprocesses the data, trains a Random Forest Regressor, and returns predictions on the test set.

    Parameters
    ----------
    - dataframe (pandas.DataFrame): A DataFrame containing the input features and the target variable.
    - features (list): A list of column names representing the features used as input for the model.
    - target (str): The name of the column in the DataFrame that represents the target variable.

    Returns
    -------
    - tuple: A tuple containing two elements:
        - y_test (numpy.ndarray): The true target values from the test set.
        - pred (numpy.ndarray): The predicted target values from the test set.

    The function performs the following steps:
    1. Copies the input 'dataframe' to a new variable 'self_dataframe' for internal use.
    2. Initializes a MinMaxScaler to scale the features between -1 and 1.
    3. Iterates through the features (excluding the target variable), scaling each feature using the MinMaxScaler.
    4. Converts the preprocessed DataFrame into a NumPy array:
       - 'X' contains the scaled feature data (excluding the target).
       - 'Y' contains the target variable data.
    5. Splits the data into training and test sets (70% for training, 30% for testing), using a fixed random seed (7) for reproducibility.
    6. Initializes a Random Forest Regressor with 100 decision trees and a fixed random seed.
    7. Trains the model using the training data (X_train and y_train).
    8. Uses the trained model to predict the target values for the test set (X_test).
    9. Returns the actual test target values (y_test) and the predicted values (pred) for evaluation.
    """
    
    # Assign 'dataframe' to 'self_dataframe' to work with a new variable.
    self_dataframe = dataframe

    # Initialize a MinMaxScaler to standardize features to a range between -1 and 1.
    sc = MinMaxScaler(feature_range=(-1, 1))

    # Define the target variable for later use.
    self_target = target

    # List the features (excluding the target) that will be standardized.
    self_features = features

    # Iterate over each feature in the list.
    for var in self_features:
        # Standardize each feature to the range [-1, 1] using MinMaxScaler.
        # Reshape the data to be 2D, as required by the scaler.
        # Skip the target variable from standardization.
        if(var != self_target):
            self_dataframe[var] = sc.fit_transform(self_dataframe[var].values.reshape(-1, 1))

    # Convert the DataFrame to a NumPy array for model compatibility, removing labels.
    # First, drop the target column and convert the remaining features to a NumPy array.
    X = self_dataframe.drop(columns=self_target).to_numpy()

    # Convert the target variable to a NumPy array.
    Y = dataframe[self_target].to_numpy()

    # Set a random seed for reproducibility of the results.
    seed = 7
    np.random.seed(seed)

    # Split the dataset into training and testing sets.
    # 70% of the data is used for training, and 30% is used for testing.
    # This splits the data into X_train, X_test, y_train, and y_test.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
    
    # Initialize the Random Forest Regressor with 100 trees and a fixed random seed for reproducibility
    regressor = RandomForestRegressor(n_estimators=100, random_state=seed)

    # Fit the model to the training data (X_train as input features, y_train as target values)
    regressor.fit(X_train, y_train)

    # Predict on the test data (X_test contains the input features, pred stores the predicted values)
    pred = regressor.predict(X_test)
    
    return y_test, pred