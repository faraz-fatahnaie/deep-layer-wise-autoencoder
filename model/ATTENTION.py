import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, \
    BatchNormalization, Attention, Flatten, Dense
from keras.models import Model

seed = np.random.seed(0)


class Attention1D(Model):
    def __init__(self, in_shape: tuple = (118, 1), out_shape: int = 2):
        super(Attention1D, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        if self.out_shape == 1:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'

        query_input = Input(shape=self.in_shape, dtype='float32')

        cnn_layer = Conv1D(filters=64, kernel_size=64, strides=1, padding='same', activation='relu')(query_input)
        pool = MaxPooling1D(pool_size=4)(cnn_layer)
        norm = BatchNormalization()(pool)
        attention = Attention()([norm, norm])

        cnn_layer2 = Conv1D(filters=128, kernel_size=64, strides=1, padding='same', activation='relu')(attention)
        pool2 = MaxPooling1D(pool_size=2)(cnn_layer2)
        norm2 = BatchNormalization()(pool2)
        attention2 = Attention()([norm2, norm2])

        cnn_layer3 = Conv1D(filters=256, kernel_size=64, strides=1, padding='same', activation='relu')(attention2)
        pool3 = MaxPooling1D(pool_size=2)(cnn_layer3)
        norm3 = BatchNormalization()(pool3)
        attention3 = Attention()([norm3, norm3])

        flatten = Flatten()(attention3)
        output = Dense(self.out_shape,
                       activation=self.activation,
                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                       bias_regularizer=tf.keras.regularizers.L2(1e-4),
                       activity_regularizer=tf.keras.regularizers.L2(1e-5))(flatten)

        self.model = Model(inputs=query_input, outputs=output)

        self.initialize()

    def call(self, inputs):
        return self.model(inputs)

    def initialize(self):
        for layer in self.model.layers:
            if isinstance(layer, (Conv1D, Dense)):
                layer.kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed)
                layer.bias_initializer = tf.keras.initializers.Zeros()

    def build_graph(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    cf = Attention1D((118, 1))
    cf.build_graph()
