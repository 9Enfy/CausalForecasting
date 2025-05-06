import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.layers import Conv1D, BatchNormalization, Flatten, Dropout, Dense
from matplotlib import pyplot as plt
import keras
from keras import models, Sequential
from keras import layers
from keras import metrics

from sklearn.preprocessing import StandardScaler
from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error

def create_model(size,X_train):

    WINDOW_SIZE = 168
    
    model = Sequential()
    # 1D Convolutional layers with dilations
    model.add(Conv1D(filters=1, kernel_size=X_train.shape[1], activation='relu', dilation_rate=1, padding='causal', input_shape=(None,1)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=1, kernel_size=X_train.shape[1], activation='relu', dilation_rate=2, padding='causal'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=1, kernel_size=X_train.shape[1], activation='relu', dilation_rate=4, padding='causal'))
    #model.add(Flatten())
    #model.add(Dropout(0.2))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(1))  # Predict next load value
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.mean_absolute_error])

    return model

def TCN(dataframe,features,target):
    # Assign 'df' to 'dataframe' to work with a new variable.
    self_dataframe = dataframe

    # Initialize a MinMaxScaler to standardize features to a range between -1 and 1.
    sc = MinMaxScaler(feature_range=(-1, 1))

    # Define the target variable, which is 'cooling_demand', for later use.
    self_target = target

    # List the features (excluding the target) that will be standardized.
    self_features = features

    num_features = len(features)

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

    # Further split the training set into training and validation sets.
    # 80% of the training data is used for training, and 20% is used for validation.
    # This helps in tuning hyperparameters and assessing model performance during training.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    
   

    # Create a model using the 'create_model_large' function.
    # This function defines the architecture of the model, which is set to be large in this case.
    model = create_model(num_features, X_train)

    # Train the model using the training data and validate it with the validation data.
    # - 'X_train' and 'y_train' are the training features and labels respectively.
    # - 'epochs=150' specifies that the model will be trained for 150 epochs, or complete passes through the training dataset.
    # - 'batch_size=32' defines the number of samples processed before the model's weights are updated.
    # - 'validation_data=(X_val, y_val)' provides validation data to evaluate the model's performance after each epoch.
    #   This helps in monitoring the model's performance on unseen data and can help in early stopping to prevent overfitting.
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val))

    # Make predictions using the test data.
    # - 'model.predict(X_test)' generates predictions for the test set.
    # - '.reshape(1, -1)[0]' reshapes the prediction array to ensure it is a one-dimensional array, suitable for comparison with 'y_test'.
    pred = model.predict(X_test).reshape(1, -1)[0]

    return y_test, pred

