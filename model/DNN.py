import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

seed = np.random.seed(0)


class DNN(Model):
    def __init__(self, in_shape: tuple = (118,), out_shape: int = 2):
        super(DNN, self).__init__()
        self.in_shape = in_shape
        if out_shape == 1:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'
        query_input = Input(shape=self.in_shape, dtype='float32')
        out = Dense(128,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=tf.keras.regularizers.L2(1e-4),
                    activity_regularizer=tf.keras.regularizers.L2(1e-5))(query_input)
        out = Dense(out_shape,
                    activation=self.activation,
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=tf.keras.regularizers.L2(1e-4),
                    activity_regularizer=tf.keras.regularizers.L2(1e-5))(out)
        self.model = Model(inputs=query_input, outputs=out)

        self.initialize()
        self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

    def initialize(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed)
                layer.bias_initializer = tf.keras.initializers.Zeros()

    def build_graph(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    cf = DNN(in_shape=(118,), out_shape=2).build_graph()
