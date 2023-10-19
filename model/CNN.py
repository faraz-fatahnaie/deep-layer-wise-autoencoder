from tensorflow import keras
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation


class CNN_MAGNETO(keras.Model):
    def __init__(self, in_shape: tuple = (118, 1), out_shape: int = 2):
        super(CNN_MAGNETO, self).__init__()
        self.in_shape = in_shape
        if out_shape == 1:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'

        self.conv0 = Conv1D(32, 2, strides=1, activation='relu', input_shape=(118, 1))
        self.conv1 = Conv1D(64, 2, strides=1, activation='relu')
        self.conv2 = Conv1D(128, 2, strides=1, activation='relu')

        self.flatten = Flatten()
        self.dense0 = Dense(256, activation='relu')
        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(out_shape, activation=self.activation)

    def call(self, inputs):
        y = self.conv0(inputs)
        y = Dropout(0.1)(y)

        y = self.conv1(y)
        y = Dropout(0.1)(y)

        y = self.conv2(y)

        y = self.flatten(y)
        y = self.dense0(y)
        y = self.dense1(y)
        y = self.dense2(y)

        return y

    def build_graph(self):
        x = keras.Input(shape=self.in_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
