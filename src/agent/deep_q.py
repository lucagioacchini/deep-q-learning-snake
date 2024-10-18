from keras import Sequential
from keras.optimizers import Adam # type: ignore
from keras.layers import Dense, Dropout, Activation # type: ignore

class DeepQNetwork():
    """ Neural Network used in Deep-Q-Learning

    Attributes:
        model (keras.Sequential): neural network model

    Methods:
        create_model(): initialize and compile the keras model
    """
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        """Initialize and compile the keras model

        Returns:
            keras.Sequential: the compiled model
        """
        model = Sequential()
        model.add(Dense(256, input_shape=(11,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4))

        model.compile(loss='mse', optimizer=Adam(
            learning_rate=1E-3), metrics=['accuracy'])
        
        model.summary()

        return model