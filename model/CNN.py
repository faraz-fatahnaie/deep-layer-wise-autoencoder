import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

seed = np.random.seed(0)


class CnnMagneto(keras.Model):
    def __init__(self, in_shape: tuple = (118, 1), out_shape: int = 2):
        super(CnnMagneto, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        if self.out_shape == 1:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'

        self.model = keras.Sequential()
        # =====================
        # self.model.add(Conv1D(32, 2, strides=1, activation='relu', input_shape=(118, 1)))
        # self.model.add(Conv1D(64, 2, strides=1, activation='relu'))
        # self.model.add(Conv1D(128, 2, strides=1, activation='relu'))
        # =====================
        # self.model.add(Conv1D(8, 64, strides=1, activation='relu', input_shape=in_shape))
        # self.model.add(Conv1D(16, 32, strides=1, activation='relu'))
        # self.model.add(Conv1D(32, 16, strides=1, activation='relu'))
        # self.model.add(Conv1D(64, 8, strides=1, activation='relu'))
        # =====================
        # self.model.add(Conv1D(8, 32, strides=1, activation='relu', input_shape=in_shape))
        # self.model.add(Conv1D(16, 16, strides=1, activation='relu'))
        # self.model.add(Conv1D(32, 8, strides=1, activation='relu'))
        # self.model.add(Conv1D(64, 4, strides=1, activation='relu'))
        # =====================
        self.model.add(Conv1D(32, 32, strides=1, activation='relu', input_shape=in_shape))
        self.model.add(Conv1D(64, 16, strides=1, activation='relu'))
        self.model.add(Conv1D(128, 8, strides=1, activation='relu'))

        self.model.add(Flatten())
        # self.model.add(Dense(256, activation='relu'))
        # self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(self.out_shape,
                             activation=self.activation,
                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                             bias_regularizer=tf.keras.regularizers.L2(1e-4),
                             activity_regularizer=tf.keras.regularizers.L2(1e-5)
                             ))

        self.initialize()
        # self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

    def initialize(self):
        for layer in self.model.layers:
            if isinstance(layer, (Conv1D, Dense)):
                layer.kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed)
                layer.bias_initializer = tf.keras.initializers.Zeros()

    def build_graph(self):
        x = keras.Input(shape=self.in_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    cf = CnnMagneto(in_shape=(118, 1), out_shape=2)
    cf = cf.build_graph()
