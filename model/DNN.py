import keras.activations
import tensorflow as tf
from keras.layers import Input, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, \
    BatchNormalization, Attention, Flatten, Dense
from keras.models import Model


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

    def call(self, inputs):
        return self.model(inputs)

    def build_graph(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    cf = DNN(in_shape=(118,), out_shape=2).build_graph()
    cf.summary()
