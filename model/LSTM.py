import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dropout, Dense, Flatten, Reshape, Activation, \
    Embedding, Add, Conv2D, MaxPooling2D, ConvLSTM2D, GRU, concatenate, Concatenate
from tensorflow.keras.models import Model, Sequential


class BiLstm(Model):
    def __init__(self, in_shape: tuple = (1, 118), out_shape: int = 2):
        super(BiLstm, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        if self.out_shape == 1:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'

        # model_input = Input(shape=(1, x_train.shape[2]))
        # forward_layer = LSTM(units=params['s_unit'], return_sequences=True)
        # backward_layer = LSTM(units=params['s_unit'], return_sequences=True, go_backwards=True)
        # x_s = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='sum')(model_input)
        # x_s = Dropout(params['s_dropout'])(x_s)
        # x_s = Flatten()(x_s)
        #
        # forward_layer = LSTM(units=params['m_unit'], return_sequences=True)
        # backward_layer = LSTM(units=params['m_unit'], return_sequences=True, go_backwards=True)
        # x_m = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='mul')(model_input)
        # x_m = Dropout(params['m_dropout'])(x_m)
        # x_m = Flatten()(x_m)
        #
        # forward_layer = LSTM(units=params['c_unit'], return_sequences=True)
        # backward_layer = LSTM(units=params['c_unit'], return_sequences=True, go_backwards=True)
        # x_c = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='concat')(model_input)
        # x_c = Dropout(params['c_dropout'])(x_c)
        # x_c = Flatten()(x_c)
        #
        # x = keras.layers.Concatenate()([x_c, x_s, x_m])
        # output = Dense(y_train.shape[1],
        #                activation="softmax"
        #                )(x)
        #
        # model = tf.keras.Model(inputs=model_input, outputs=output)

        model_input = Input(shape=self.in_shape)

        forward_layer = LSTM(units=128, return_sequences=True)
        backward_layer = LSTM(units=128, return_sequences=True, go_backwards=True)
        x_concat = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='concat')(model_input)
        x = Dropout(0.5)(x_concat)
        x = Flatten()(x)

        output = Dense(self.out_shape,
                       activation="softmax",
                       # kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                       # bias_regularizer=tf.keras.regularizers.L2(1e-4),
                       # activity_regularizer=tf.keras.regularizers.L2(1e-5)
                       )(x)

        self.model = tf.keras.Model(inputs=model_input, outputs=output)

        # self.initialize()
        self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

    def initialize(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.kernel_initializer = tf.keras.initializers.GlorotNormal()
                layer.bias_initializer = tf.keras.initializers.Zeros()

    def build_graph(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class CnnLstm(Model):
    def __init__(self, in_shape: tuple = (118, 1), out_shape: int = 2):
        super(CnnLstm, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.model = Sequential()

        self.model.add(Conv1D(8, 32, activation='relu', input_shape=self.in_shape))
        # self.model.add(MaxPooling1D(2))

        self.model.add(Conv1D(16, 16, activation='relu'))
        # self.model.add(MaxPooling1D(2))

        self.model.add(Conv1D(32, 8, activation='relu'))
        # self.model.add(MaxPooling1D(2))

        self.model.add(Conv1D(64, 4, activation='relu'))
        # self.model.add(MaxPooling1D(2))

        self.model.add(Reshape((1, -1)))

        self.model.add(LSTM(50, activation='tanh', return_sequences=True))
        # self.model.add(Dropout(0.1))

        self.model.add(LSTM(25, activation='tanh'))
        # self.model.add(Dropout(0.1))

        self.model.add(Flatten())

        # self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.out_shape,
                             activation='softmax',
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
                layer.kernel_initializer = tf.keras.initializers.GlorotNormal()
                layer.bias_initializer = tf.keras.initializers.Zeros()

    def build_graph(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class ConvLstm(Model):
    def __init__(self, in_shape: tuple = (1, 1, 10, 10), out_shape: int = 2):
        super(ConvLstm, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        if self.out_shape == 1:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'

        model = Sequential()
        model.add(ConvLSTM2D(filters=32, kernel_size=(2, 2), padding='same', data_format='channels_first',
                             input_shape=self.in_shape,
                             return_sequences=True))
        model.add(Dropout(0.2))
        model.add(ConvLSTM2D(filters=64, kernel_size=(2, 2), padding='same', data_format='channels_first',
                             return_sequences=True))
        model.add(Dropout(0.2))
        model.add(ConvLSTM2D(filters=128, kernel_size=(2, 2), padding='same', data_format='channels_first',
                             return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        self.model = model
        # print(self.model.summary())

    def call(self, inputs):
        return self.model(inputs)

    def build_graph(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    # cf = CnnLstm((118, 1))
    cf = BiLstm((1, 196), out_shape=2)
    cf = cf.build_graph()
