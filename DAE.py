import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, Convolution2D, Activation, MaxPooling2D, \
    Dense, BatchNormalization, Dropout
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.layers import BatchNormalization
from model.CNNATTENTION import CNNATTENTION1D
from keras.layers import Input, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, \
    BatchNormalization, Attention, Flatten, Dense
from keras.models import Model
from utils import parse_data
from model.SP import SP
from model.CNN import CNN_MAGNETO

print(keras.__version__)

# Generate synthetic data for testing (replace with your dataset)
train = pd.read_csv('C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\KDD_CUP99\\train_binary.csv')
test = pd.read_csv('C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\KDD_CUP99\\test_binary.csv')
seed = np.random.seed(0)

X_train, y_train = parse_data(train, 'KDD_CUP99', 'binary')
X_test, y_test = parse_data(test, 'KDD_CUP99', 'binary')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

pretraining_epochs = 10
pretraining_batch_size = 100
initial_learning_rate = 0.1
final_learning_rate = 0.00001
pretraining_learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / pretraining_epochs)
pretraining_steps_per_epoch = int(len(X_train) / pretraining_batch_size)

input_img = Input(shape=(X_train.shape[1],))
distorted_input1 = Dropout(.1)(input_img)
encoded1 = Dense(32, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(
    distorted_input1)
encoded1_bn = BatchNormalization()(encoded1)
decoded1 = Dense(X_train.shape[1], activation='sigmoid',
                 kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(encoded1_bn)

autoencoder1 = Model(inputs=input_img, outputs=decoded1)
encoder1 = Model(inputs=input_img, outputs=encoded1_bn)

# Layer 2
encoded1_input = Input(shape=(32,))
distorted_input2 = Dropout(.2)(encoded1_input)
encoded2 = Dense(32, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(
    distorted_input2)
encoded2_bn = BatchNormalization()(encoded2)
decoded2 = Dense(32, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(
    encoded2_bn)

autoencoder2 = Model(inputs=encoded1_input, outputs=decoded2)
encoder2 = Model(inputs=encoded1_input, outputs=encoded2_bn)

# Layer 3 - which we won't end up fitting in the interest of time
encoded2_input = Input(shape=(32,))
distorted_input3 = Dropout(.3)(encoded2_input)
encoded3 = Dense(32, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(
    distorted_input3)
encoded3_bn = BatchNormalization()(encoded3)
decoded3 = Dense(32, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(
    encoded3_bn)

autoencoder3 = Model(inputs=encoded2_input, outputs=decoded3)
encoder3 = Model(inputs=encoded2_input, outputs=encoded3_bn)

# Deep Autoencoder
encoded1_da = Dense(32, activation='sigmoid')(input_img)
encoded1_da_bn = BatchNormalization()(encoded1_da)
encoded2_da = Dense(32, activation='sigmoid')(encoded1_da_bn)
encoded2_da_bn = BatchNormalization()(encoded2_da)
encoded3_da = Dense(32, activation='sigmoid')(encoded2_da_bn)
encoded3_da_bn = BatchNormalization()(encoded3_da)
decoded3_da = Dense(32, activation='sigmoid')(encoded3_da_bn)
decoded2_da = Dense(32, activation='sigmoid')(decoded3_da)
decoded1_da = Dense(X_train.shape[1], activation='sigmoid')(decoded2_da)

deep_autoencoder = Model(inputs=input_img, outputs=decoded1_da)

# Pretraining step
sgd1 = SGD(learning_rate=5, decay=0.5, momentum=.85, nesterov=True)
sgd2 = SGD(learning_rate=5, decay=0.5, momentum=.85, nesterov=True)
sgd3 = SGD(learning_rate=5, decay=0.5, momentum=.85, nesterov=True)

lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=pretraining_steps_per_epoch,
    decay_rate=pretraining_learning_rate_decay_factor,
    staircase=True)
opt1 = Adam(lr_schedule1)
lr_schedule2 = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=pretraining_steps_per_epoch,
    decay_rate=pretraining_learning_rate_decay_factor,
    staircase=True)
opt2 = Adam(lr_schedule2)
lr_schedule3 = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=pretraining_steps_per_epoch,
    decay_rate=pretraining_learning_rate_decay_factor,
    staircase=True)
opt3 = Adam(lr_schedule3)

autoencoder1.compile(loss='mae', optimizer=sgd1)
autoencoder2.compile(loss='mae', optimizer=sgd2)
autoencoder3.compile(loss='mae', optimizer=sgd3)

encoder1.compile(loss='mae', optimizer=sgd1)
encoder2.compile(loss='mae', optimizer=sgd2)
encoder3.compile(loss='mae', optimizer=sgd3)

deep_autoencoder.compile(loss='mae', optimizer=sgd1)

# Layer-wise training
autoencoder1.fit(X_train, X_train,
                 epochs=pretraining_epochs, batch_size=pretraining_batch_size,
                 # validation_split=0.20,
                 shuffle=True)

first_layer_code = encoder1.predict(X_train)
print(first_layer_code.shape)
autoencoder2.fit(first_layer_code, first_layer_code,
                 epochs=pretraining_epochs, batch_size=pretraining_batch_size,
                 # validation_split=0.20,
                 shuffle=True)

second_layer_code = encoder2.predict(first_layer_code)
print(second_layer_code.shape)
autoencoder3.fit(second_layer_code, second_layer_code,
                 epochs=pretraining_epochs, batch_size=pretraining_batch_size,
                 # validation_split=0.20,
                 shuffle=True)

# Finetune step

# Setting the weights of the deep autoencoder
deep_autoencoder.layers[1].set_weights(autoencoder1.layers[2].get_weights())  # first dense layer
deep_autoencoder.layers[2].set_weights(autoencoder1.layers[3].get_weights())  # first bn layer
deep_autoencoder.layers[3].set_weights(autoencoder2.layers[2].get_weights())  # second dense layer
deep_autoencoder.layers[4].set_weights(autoencoder2.layers[3].get_weights())  # second bn layer
deep_autoencoder.layers[5].set_weights(autoencoder3.layers[2].get_weights())  # third dense layer
deep_autoencoder.layers[6].set_weights(autoencoder3.layers[3].get_weights())  # third bn layer
deep_autoencoder.layers[7].set_weights(autoencoder3.layers[4].get_weights())  # first decoder
deep_autoencoder.layers[8].set_weights(autoencoder2.layers[4].get_weights())  # second decoder
deep_autoencoder.layers[9].set_weights(autoencoder1.layers[4].get_weights())  # third decoder

# dense1 = Dense(128, activation='relu')(decoded1_da)
# dense = Dense(1, activation='sigmoid')(dense1)
# classifier = Model(inputs=input_img, outputs=dense)

# attention = CNNATTENTION1D().build_graph(raw_shape=(118, 1))
# out = attention(decoded1_da)
# classifier = Model(inputs=input_img, outputs=out)

cnn = CNN_MAGNETO().build_graph(raw_shape=(118, 1))
out = cnn(decoded1_da)
classifier = Model(inputs=input_img, outputs=out)

# sp = SP(classification_mode='binary')
# decoded1_da = tf.keras.layers.Reshape((14, 14))(decoded1_da)
# sp_out = sp(decoded1_da)
# classifier = Model(inputs=input_img, outputs=sp_out)

sgd4 = SGD(learning_rate=.1, decay=0.001, momentum=.95, nesterov=True)

finetune_epochs = 10
finetune_batch_size = 100
initial_learning_rate = 0.01
final_learning_rate = 0.00001
finetune_learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / finetune_epochs)
finetune_steps_per_epoch = int(len(X_train) / finetune_batch_size)

lr_schedule4 = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=finetune_steps_per_epoch,
    decay_rate=finetune_learning_rate_decay_factor,
    staircase=True)
opt4 = Adam(lr_schedule4)

# for layer in classifier.layers[:-1]:
#     layer.trainable = False

classifier.compile(loss='binary_crossentropy', optimizer=opt4, metrics=['accuracy'])

classifier.fit(X_train, y_train, validation_data=(X_test, y_test),
               epochs=finetune_epochs, batch_size=finetune_batch_size,
               shuffle=True)

# for layer in classifier.layers[:]:
#     layer.trainable = True
#
# lr_schedule5 = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=finetune_steps_per_epoch,
#     decay_rate=finetune_learning_rate_decay_factor,
#     staircase=True)
# opt5 = Adam(lr_schedule5)
#
# classifier.compile(loss='binary_crossentropy', optimizer=opt5, metrics=['accuracy'])
#
# classifier.fit(X_train, y_train, validation_data=(X_test, y_test),
#                epochs=finetune_epochs, batch_size=finetune_batch_size,
#                shuffle=True)
