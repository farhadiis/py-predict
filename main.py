import pandas
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot
from numpy import concatenate


def start():
    print('Start app...')
    # Load CSV data into a dataframe
    dataframe = pandas.read_csv('gm.csv', index_col='date')
    # Add to predict column (adjusted close) and shift it. This is our output
    dataframe['output'] = dataframe.adjusted_close.shift(-1)
    # Remove NaN on the final sample (because we don't have tomorrow's output)
    dataframe = dataframe.dropna()

    # Rescale
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    rescaled = scaler.fit_transform(dataframe.values)

    # Split into training/testing
    training_ratio = 0.8
    training_testing_index = int(len(rescaled) * training_ratio)
    training_data = rescaled[:training_testing_index]
    testing_data = rescaled[training_testing_index:]
    training_length = len(training_data)
    testing_length = len(testing_data)

    # Split training into input/output. Output is the one we added to the end
    training_input_data = training_data[:, 0:-1]
    training_output_data = training_data[:, -1]

    # Split testing into input/output. Output is the one we added to the end
    testing_input_data = testing_data[:, 0:-1]
    testing_output_data = testing_data[:, -1]

    # Reshape data for (Sample, Timesteps, Features)
    training_input_data = training_input_data.reshape(training_input_data.shape[0], 1, training_input_data.shape[1])
    testing_input_data = testing_input_data.reshape(testing_input_data.shape[0], 1, testing_input_data.shape[1])

    # Build the model
    model = Sequential()
    model.add(LSTM(100, input_shape=(training_input_data.shape[1], training_input_data.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit model with history to check for overfitting
    history = model.fit(
        training_input_data,
        training_output_data,
        epochs=100,
        validation_data=(testing_input_data, testing_output_data),
        shuffle=False
    )

    pyplot.plot(history.history['loss'], label='Training Loss')
    pyplot.plot(history.history['val_loss'], label='Testing Loss')
    pyplot.legend()
    pyplot.show()

    # Generate predictions
    raw_predictions = model.predict(testing_input_data)

    # Reshape testing input data back to 2d
    testing_input_data = testing_input_data.reshape((testing_input_data.shape[0], testing_input_data.shape[2]))
    testing_output_data = testing_output_data.reshape((len(testing_output_data), 1))

    # Invert scaling for prediction data
    unscaled_predictions = concatenate((testing_input_data, raw_predictions), axis=1)
    unscaled_predictions = scaler.inverse_transform(unscaled_predictions)
    unscaled_predictions = unscaled_predictions[:, -1]

    # Invert scaling for actual data
    unscaled_actual_data = concatenate((testing_input_data, testing_output_data), axis=1)
    unscaled_actual_data = scaler.inverse_transform(unscaled_actual_data)
    unscaled_actual_data = unscaled_actual_data[:, -1]

    # Plot prediction vs actual
    pyplot.plot(unscaled_actual_data, label='Actual Adjusted Close')
    pyplot.plot(unscaled_predictions, label='Predicted Adjusted Close')
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    start()
