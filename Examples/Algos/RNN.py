import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras import metrics
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error

def create_sequences(X, y, time_steps):
    r"""Define a function to create sequences of data for time series forecasting.
     - X: Input features (e.g., sensor readings, time series data).
     - y: Target values (e.g., values you want to predict).
     - time_steps: The number of time steps to include in each input sequence.
    """
    Xs, ys = [], []  # Initialize empty lists to hold the input sequences and corresponding target values.
    
    # Loop through the data to create sequences.
    for i in range(len(X) - time_steps):
        # Append a sequence of features from index i to i + time_steps to the list Xs.
        Xs.append(X[i:i + time_steps])
        
        # Append the target value that corresponds to the end of the sequence to the list ys.
        ys.append(y[i + time_steps])
    
    # Convert the lists of sequences and target values to NumPy arrays.
    return np.array(Xs), np.array(ys)

def create_model(time_steps, X_train, size):
    r"""
    Creates and returns a recurrent neural network (RNN) model using SimpleRNN layers, 
    designed for time series or sequential data prediction.

    Parameters
    ----------
    - time_steps (int): The number of time steps in each input sequence, representing 
                        the temporal length of the data sequences.
    - X_train (numpy.ndarray): The training feature data, used to define the input shape of the model. 
                               Specifically, the shape of the third dimension, X_train.shape[2], 
                               represents the number of features in each time step.
    - size (int): The base number of neurons (units) used in the SimpleRNN layers. It controls 
                  the model's capacity and complexity.

    Returns
    ---------
    - model (keras.Sequential): A Sequential RNN model ready to be compiled and trained.

    The function performs the following operations:
    1. Initializes a Sequential model, which allows for a linear stacking of layers.
    2. Adds the first SimpleRNN layer:
       - Uses 'size' units (neurons) and ReLU activation.
       - The input shape is defined by 'time_steps' (the number of time steps in the sequence) 
         and 'X_train.shape[2]' (the number of features).
       - 'return_sequences=True' ensures that the full sequence of outputs is returned, allowing 
         this output to be passed to the next RNN layer.
    3. Adds the second SimpleRNN layer:
       - Uses 'size * 2' units and ReLU activation.
       - 'return_sequences=True' continues passing the full sequence of outputs to the next layer.
    4. Adds the third SimpleRNN layer:
       - Uses 'size' units and ReLU activation.
       - 'return_sequences=False' ensures that only the final time step's output is passed, suitable 
         for tasks where only the final prediction is needed (e.g., regression).
    5. Adds a Dense output layer with 1 unit (suitable for regression tasks where the output is a 
       single value).
    
    The model is returned and ready for compilation and training.
    """
    # Initialize a Sequential model, which allows for a linear stack of layers.
    model = Sequential()

    # Add the first SimpleRNN layer.
    # - size units specify the number of neurons in this RNN layer.
    # - 'activation='relu'' applies the ReLU activation function to the outputs.
    # - 'input_shape=(time_steps, X_train.shape[2])' defines the shape of the input data: 
    #   'time_steps' for the number of time steps in each sequence and 'X_train.shape[2]' for the number of features.
    # - 'return_sequences=True' ensures that this layer returns the full sequence of outputs for each time step, 
    #   which is required as input for the subsequent RNN layer.
    model.add(SimpleRNN(size, activation='relu', input_shape=(time_steps, X_train.shape[2]), return_sequences=True))

    # Add the second SimpleRNN layer.
    # - size*2 units specify the number of neurons in this RNN layer.
    # - 'activation='relu'' applies the ReLU activation function.
    # - 'return_sequences=True' ensures that this layer also returns the full sequence of outputs.
    model.add(SimpleRNN(size*2, activation='relu', return_sequences=True))

    # Add the third SimpleRNN layer.
    # - size units specify the number of neurons in this RNN layer.
    # - 'activation='relu'' applies the ReLU activation function.
    # - 'return_sequences=False' means this layer will return only the output for the final time step,
    #   which is suitable for producing a single prediction.
    model.add(SimpleRNN(size, activation='relu', return_sequences=False))

    # Add a Dense layer to output the prediction.
    # - The Dense layer has 1 unit, which is suitable for regression tasks where the output is a single value.
    model.add(Dense(1))

    return model


def RNN(dataframe, target, time_steps):
    r"""
    Builds, trains, and evaluates a recurrent neural network (RNN) for time series prediction.

    Parameters
    ----------
    - dataframe (pandas.DataFrame): A DataFrame containing the input features and the target variable.
    - target (str): The name of the column that represents the target variable to be predicted (e.g., 'cooling_demand').
    - time_steps (int): The number of time steps to use when creating sequences for the RNN model.
    
    Returns
    -------
    - tuple: A tuple containing:
        - y_test (numpy.ndarray): The true target values from the test set.
        - pred (numpy.ndarray): The predicted target values from the test set.

    The function performs the following steps:
    1. Initializes two MinMaxScaler objects:
       - 'scaler_x' to scale the independent variables (features).
       - 'scaler_y' to scale the target variable (e.g., 'cooling_demand').
    2. Scales the independent variables (all columns except the target) using the MinMaxScaler.
    3. Scales the target variable separately.
    4. Creates sequences of data using the 'create_sequences' function, generating arrays of sequences (X_sequences) 
       and their corresponding target values (y_sequences), based on the specified number of time steps.
    5. Splits the time series data into training and testing sets, where 70% of the sequences are used for training, 
       and 30% for testing.
    6. Further splits the training set into training and validation sets, using 80% of the training data for training 
       and 20% for validation.
    7. Calls the 'create_model' function to build the RNN model, specifying:
       - The number of time steps in the input data.
       - The number of features in the dataset.
    8. Compiles the model using the Adam optimizer and the mean squared error loss function, with mean absolute error (MAE) 
       as a performance metric.
    9. Trains the model using the training data and validates it using the validation data, for 150 epochs with a batch size of 32.
    10. After training, the model generates predictions on the test set.
    11. Reshapes the predictions to a one-dimensional array and returns both the true test values (y_test) and the predictions (pred).
    """
        
    # Initialize two MinMaxScaler objects: 
    # - 'scaler_x' for scaling the independent variables (features).
    # - 'scaler_y' for scaling the target variable ('cooling_demand').
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Assign the DataFrame 'dataframe' to 'dataset' to use for scaling.
    self_dataset = dataframe

    # Scale the independent variables (all columns except target).
    # The 'fit_transform' method adjusts the scaler to the data and then transforms it.
    X_scaled = scaler_x.fit_transform(self_dataset.drop(columns=target))

    # Scale the target variable separately. 
    # The double brackets are used to select the column as a DataFrame (2D array) rather than a Series.
    y_scaled = scaler_y.fit_transform(self_dataset[[target]])
    
    # Generate sequences of data for training. 
    # The 'create_sequences' function takes the scaled features (X_scaled) and target (y_scaled),
    # and creates sequences of length 'time_steps' for both X (features) and y (target).
    # - X_sequences: Array of sequences of feature data.
    # - y_sequences: Array of corresponding target values.
    X_sequences, y_sequences = create_sequences(X_scaled, y_scaled, time_steps)

    # Set a random seed for reproducibility of the results.
    seed = 7
    np.random.seed(seed)

    # Split the time series data into training and testing sets.
    # 70% of the sequences are used for training, and 30% are used for testing.
    # This results in X_train, X_test, y_train, and y_test.
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.3, random_state=seed)

    # Further split the training set into training and validation sets.
    # 80% of the training sequences are used for training, and 20% are used for validation.
    # This helps in tuning model parameters and assessing model performance during training.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    num_features = len(self_dataset.drop(columns=target))

    model = create_model(time_steps, X_train, num_features)

    # Compile the model.
    # - 'optimizer='adam'' specifies the Adam optimizer, which adapts the learning rate during training.
    # - 'loss='mean_squared_error'' sets the loss function to mean squared error, appropriate for regression tasks.
    # - 'metrics=[metrics.mae]' includes mean absolute error as a metric to monitor during training and evaluation.
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mean_squared_error"])

    # Train the model using the training data and validate it with the validation data.
    # - 'X_train' and 'y_train' are the training features and labels respectively.
    # - 'epochs=150' specifies that the model will be trained for 150 epochs, or complete passes through the training dataset.
    # - 'batch_size=32' defines the number of samples processed before the model's weights are updated.
    # - 'validation_data=(X_val, y_val)' provides validation data to evaluate the model's performance after each epoch.
    #   This helps in monitoring the model's performance on unseen data and can help in early stopping to prevent overfitting.
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val))

    # Make predictions using the test data.
    # - 'model.predict(X_test)' generates predictions for the test set.
    # - '.reshape(1, -1)[0]' reshapes the prediction array to ensure it is a one-dimensional array, suitable for comparison with 'y_test'.
    pred = model.predict(X_test).reshape(1, -1)[0]

    
    return y_test, pred

