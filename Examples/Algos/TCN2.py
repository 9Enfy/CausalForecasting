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


def create_sliding_windows(data, window_size, forecast_horizon):
    """
    Divide a dataset in multiple dataset

    Parameters
    ----------
    - data (numpy.ndarray): The dataset to divide.
    - window_size (int): number of hours used in a singular prediction. A bigger window_size  
        SHOULD mean a more accurate singuar prediction, however the number of prediction made
        will be lower. If it is too high the function will fail
    - forecast_horizon (int): number of hours to predict.

    Returns
    ----------
    - X (numpy.ndarray): A matrix with the divided dataset
    - y (numpy.ndarray): A matrix with the divided dataset + a space reserved for the prediction

    The function consist in a cycle to populate the X and y matrix.
    As an example data is an array containing the numbers from 1 to 10 inclusive, 
    the window_size is equals to 3 and forecast_horizon equals to 2.
    The first index of the X array will contain an array containing 1-2-3. The second index will contain
    an array containing 2-3-4, the third will contain 3-4-5 and so on.
    The first index of the y array will contain an array containing 1-2-3-4-5, 
    the second index will cotain an array containing 2-3-4-5-6, the third will contain 3-4-5-6-7 and so on.
    """
    X, y = [], []
    print(type(data))
    for i in range(len(data) - window_size - forecast_horizon):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size + forecast_horizon])
    return np.array(X), np.array(y)



def create_model(size,_filters,_kernel_size,window_size):
    '''
    Creates and compiles a large artificial neural network model for temporal Convolutional Network.

    Parameters
    ----------
    - size (int): The number of neurons in the last hidden layer. 
                  This determines the complexity and capacity of the model.
    - _filters (int): the dimension of the output space (the number of filters in the convolution)
    - _kernel_size (int): specifying the size of the convolution window
    - window_size: number of hours used in a singular prediction

    Returns
    ----------
    - model (keras.Sequential): A compiled Sequential model ready for training.

    The function performs the following operations:
    1. Initializes a Sequential model, allowing layers to be stacked sequentially.
    2. Adds an input layer of type Conv1D. It has ReLU (Rectified Linear Unit) as the activation function. 
        The input shape is set to the window_size. 
        TODO: add functionality to change the padding.
    3. Adds a hidden layer for batch normalization (maintains the mean output close to 0 and the output standard deviation close to 1)
    4. Adds an input layer of type Conv1d. Same as step 2 but without input shape and a different dilation rate.
    5. Adds a hidden layer for batch normalization
    6. Repeat step 4 and 5 with a different dilation rate (this step can be repeated other times but a total of 3 times is good enough.
    
    7. Add a flatten layer. This is important to ensure compatibility between Conv1D layer and Dense layer.
    8. Add a Dropout layer (may be changed/removed?)
    9. Add a hidden layer with size neurons.
    10. Adds an output layer with a single neuron (suitable for regression tasks) and no 
       activation function.
    6. Compiles the model using the Adam optimizer, with mean squared error as the loss 
       function and mean absolute error (MAE) as a performance metric.

    The compiled model is then returned for training.
    '''
    # Model Architecture: Temporal Convolutional Network
    model = Sequential()
    model.add(Conv1D(filters=_filters, kernel_size=_kernel_size, dilation_rate=1, padding='causal', activation='relu', input_shape=(window_size, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=_filters, kernel_size=_kernel_size, dilation_rate=3, padding='causal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=_filters, kernel_size=_kernel_size, dilation_rate=9, padding='causal', activation='relu'))
    model.add(BatchNormalization())
    #model.add(Conv1D(filters=_filters, kernel_size=_kernel_size, dilation_rate=8, padding='causal', activation='relu'))
    #model.add(BatchNormalization())

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(size, activation='relu'))
    model.add(Dense(1))  # Predict next load value

    # Compile the model
    #model.compile(optimizer='adam', loss='mean_squared_error')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mean_absolute_error"])

    


    return model

def TCN(dataframe,features,target,filters=64,kernel_size=3,WINDOW_SIZE=168,FORECAST_HORIZON=24):
    '''
    Constructs and trains an temporal convolution network (TCN) to predict a specific target 
    based on a provided set of features.

    Parameters
    ----------
    - dataframe (pandas.DataFrame): A DataFrame containing the data, including both features and the target.
    - features (list): A list of column names representing the features used as inputs to the neural network.
    - target (str): The name of the column in the DataFrame that represents the target variable to be predicted.
    - filters (int, default=64): the dimension of the output space (the number of filters in the convolution)
    - kernel_size (int default=3): specifying the size of the convolution window
    - WINDOW_SIZE (int, default=168): number of hours used in a singular prediction. A bigger window_size  
        SHOULD mean a more accurate singuar prediction, however the number of prediction made
        will be lower. If it is too high (window_size+forecast_horizon> size(dataframe)) the function will fail
    - forecast_horizon (int): number of hours to predict.
    '''
     # Set a random seed for reproducibility of the results.
    seed = 7
    np.random.seed(seed)
    
    # Assign 'df' to 'dataframe' to work with a new variable.
    X, y = create_sliding_windows(dataframe[target].values, WINDOW_SIZE, FORECAST_HORIZON)

    print(y[0])
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Further split the training set into training and validation sets.
    # 80% of the training data is used for training, and 20% is used for validation.
    # This helps in tuning hyperparameters and assessing model performance during training.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    
   

    # Create a model using the 'create_model_large' function.
    # This function defines the architecture of the model, which is set to be large in this case.
    num_features = len(features)
    model = create_model(100, filters, kernel_size,WINDOW_SIZE)

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

    # Training the model with early stopping and learning rate reduction
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val),callbacks=[early_stopping, reduce_lr])
    #history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val))


    # Make predictions using the test data.
    # - 'model.predict(X_test)' generates predictions for the test set.
    # - '.reshape(1, -1)[0]' reshapes the prediction array to ensure it is a one-dimensional array, suitable for comparison with 'y_test'.
    pred = model.predict(X_test).reshape(1, -1)[0]

    return y_test, pred

