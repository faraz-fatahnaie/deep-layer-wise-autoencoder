import tensorflow as tf
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D, Dense, multiply, Flatten
from keras.models import Model


class SE(Model):

    def __init__(self, in_shape: tuple = (118, 1), out_shape: int = 2):
        super(SE, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        inputs = tf.keras.Input(shape=self.in_shape)
        x = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(filters=32, kernel_size=2, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = MaxPooling2D(pool_size=2)(x)

        b, f, ch = x.shape
        y = GlobalAveragePooling1D()(x)
        y = Dense(ch // 8, activation='relu')(y)
        y = Dense(ch, activation='sigmoid')(y)

        y = multiply([y, x])

        # print(y.shape)
        y = Flatten()(y)
        # print(y.shape)
        y = Dense(128)(y)
        output = Dense(self.out_shape, activation='softmax')(y)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        self.model = model

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


if __name__ == "__main__":
    cf = SE(in_shape=(118, 1), out_shape=2).build_graph()
