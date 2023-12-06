from keras.layers import Input, Dense, BatchNormalization, Dropout, LSTM
from keras.models import Model, Sequential
import tensorflow as tf


def ae_factory(in_shape, hidden_size, activation):
    autoencoder = Sequential()
    autoencoder.add(Input(shape=(in_shape,)))
    for size in hidden_size:
        autoencoder.add(Dense(size, activation=activation))
    return autoencoder


def partial_ae_factory(in_shape, hidden_size, activation):
    input_img = Input(shape=(in_shape,))
    encoded = Dense(hidden_size, activation=activation,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0))(input_img)
    decoded = Dense(in_shape, activation=activation,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0))(encoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoded)

    return autoencoder, encoder


# def ae_factory(in_shape, hidden_size, activation):
#     autoencoder = Sequential()
#     autoencoder.add(Input(shape=(in_shape,)))
#     # ENCODER
#     for size in hidden_size:
#         autoencoder.add(Dense(size, activation=activation))
#         autoencoder.add(BatchNormalization())
#
#     # DECODER
#     if len(hidden_size) >= 1:
#         hidden_size.pop()
#         rev_hidden_size = hidden_size[::-1]
#         if len(rev_hidden_size) > 0:
#             for size in rev_hidden_size:
#                 autoencoder.add(Dense(size, activation=activation))
#
#     autoencoder.add(Dense(in_shape, activation=activation))
#     return autoencoder
#
#
# def partial_ae_factory(in_shape, hidden_size, activation):
#     input_img = Input(shape=(in_shape,))
#     distorted_input = Dropout(0)(input_img)
#     encoded = Dense(hidden_size, activation=activation,
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0))(distorted_input)
#     encoded_bn = BatchNormalization()(encoded)
#     decoded = Dense(in_shape, activation=activation,
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0))(encoded_bn)
#
#     autoencoder = Model(inputs=input_img, outputs=decoded)
#     encoder = Model(inputs=input_img, outputs=encoded_bn)
#
#     return autoencoder, encoder


if __name__ == "__main__":
    h = [128, 64, 32]
    ae = ae_factory(in_shape=118, hidden_size=h, activation='tanh')
    ae.summary()
