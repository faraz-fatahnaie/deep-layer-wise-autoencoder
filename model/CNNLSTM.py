import tensorflow as tf
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dropout, Dense, Flatten, Reshape
from keras.models import Model, Sequential


class CNNLSTM(Model):
    def __init__(self, in_shape: tuple = (118, 1), out_shape: int = 2):
        super(CNNLSTM, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.model = Sequential()

        self.model.add(Conv1D(8, 32, activation='relu', input_shape=self.in_shape))
        self.model.add(MaxPooling1D(2))

        self.model.add(Conv1D(16, 16, activation='relu'))
        self.model.add(MaxPooling1D(2))

        self.model.add(Conv1D(32, 8, activation='relu'))
        # self.model.add(MaxPooling1D(2))

        self.model.add(Conv1D(64, 4, activation='relu'))
        # self.model.add(MaxPooling1D(2))

        self.model.add(Reshape((1, -1)))

        self.model.add(LSTM(50, return_sequences=True))
        # self.model.add(Dropout(0.1))

        self.model.add(LSTM(25))
        # self.model.add(Dropout(0.1))

        self.model.add(Flatten())

        # self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.out_shape,
                             activation='softmax',
                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                             bias_regularizer=tf.keras.regularizers.L2(1e-4),
                             activity_regularizer=tf.keras.regularizers.L2(1e-5)
                             ))

        self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

    def build_graph(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    cf = CNNLSTM((118, 1))
    cf = cf.build_graph()
