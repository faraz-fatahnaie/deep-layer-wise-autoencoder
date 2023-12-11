from __future__ import print_function

import os

import numpy as np
import pandas as pd
import keras
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
from keras import callbacks
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import np_utils
import tensorflow as tf
from keras.optimizers import Adam

from configs.setting import setting
from utils import parse_data, OptimizerFactory

np.random.seed(0)


def data(dataset_name):
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    train = pd.read_csv(
        f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}'
        f'\\train_binary.csv')
    test = pd.read_csv(
        f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}'
        f'\\test_binary.csv')

    X_train, y_train = parse_data(train, dataset_name, 'binary')
    X_test, y_test = parse_data(test, dataset_name, 'binary')
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test, train.columns, test.columns


def getBatchSize(p, bs):
    return bs[p]


def Autoencoder(X_train):
    print('START MAGNETO AUTOENCODER TRAINING ...')
    input_ae = Input((X_train.shape[1],))

    encoded = Dense(128, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod1')(input_ae)
    encoded = Dense(64, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod2')(encoded)
    encoded = Dropout(0)(encoded)
    decoded = Dense(128, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='decoder1')(encoded)
    decoded = Dense(X_train.shape[1], activation='linear',
                    kernel_initializer='glorot_uniform',
                    name='decoder3')(decoded)

    model = Model(inputs=input_ae, outputs=decoded)
    model.summary()

    adam = Adam(lr=0.001)
    model.compile(loss='mse',
                  optimizer=adam)
    model.summary()
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss',
                                min_delta=0.0001,
                                patience=10,
                                restore_best_weights=True),
    ]

    history = model.fit(X_train, X_train,
                        batch_size=128,
                        epochs=150,
                        verbose=2,
                        validation_split=0.2,
                        callbacks=callbacks_list)

    # get the highest validation accuracy of the training epochs
    # score = np.amin(history.history['val_loss'])
    # print('Best validation loss of epoch:', score)

    # scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    # score = min(scores)
    # print('Score', score)
    print('END OF MAGNETO AUTOENCODER TRAINING.')
    return model
    # return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(history.history['loss']),
    #         'n_params': model.count_params(), 'model': model}


def AutoencoderLayerWise(X_train):
    print('START TRAINING AUTOENCODER IN LAYER-WISE MODE ...')
    config, config_file = setting()
    seed = config['SEED']
    hidden_size = config['AE_HIDDEN_SIZE']

    # ================= LAYER 1 =================
    input_img = Input(shape=(X_train.shape[1],))
    encoded1 = Dense(hidden_size[0], activation=config['AE_ACTIVATION'],
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(input_img)
    decoded1 = Dense(X_train.shape[1], activation=config['AE_ACTIVATION'],
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(encoded1)

    autoencoder1 = Model(inputs=input_img, outputs=decoded1)
    encoder1 = Model(inputs=input_img, outputs=encoded1)

    # ================= LAYER 2 =================
    encoded1_input = Input(shape=(hidden_size[0],))
    encoded2 = Dense(hidden_size[1], activation=config['AE_ACTIVATION'],
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(encoded1_input)
    decoded2 = Dense(hidden_size[0], activation=config['AE_ACTIVATION'],
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(encoded2)

    autoencoder2 = Model(inputs=encoded1_input, outputs=decoded2)
    encoder2 = Model(inputs=encoded1_input, outputs=encoded2)

    # ================= LAYER 3 =================
    encoded2_input = Input(shape=(hidden_size[1],))
    encoded3 = Dense(hidden_size[2], activation=config['AE_ACTIVATION'],
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(encoded2_input)
    decoded3 = Dense(hidden_size[1], activation=config['AE_ACTIVATION'],
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(encoded3)

    autoencoder3 = Model(inputs=encoded2_input, outputs=decoded3)
    encoder3 = Model(inputs=encoded2_input, outputs=encoded3)

    # ================= AutoEncoder Layer-wise Training =================
    opt_factory_ae = OptimizerFactory(opt=config['AE_OPTIMIZER'],
                                      lr_schedule=config['AE_SCHEDULE'],
                                      len_dataset=len(X_train),
                                      epochs=config['AE_EPOCH'],
                                      batch_size=config['AE_BATCH_SIZE'],
                                      init_lr=config['AE_INITIAL_LR'],
                                      final_lr=config['AE_FINAL_LR'])
    sgd1 = opt_factory_ae.get_opt()
    sgd2 = opt_factory_ae.get_opt()
    sgd3 = opt_factory_ae.get_opt()

    autoencoder1.compile(loss=config['AE_LOSS'], optimizer=sgd1)
    autoencoder2.compile(loss=config['AE_LOSS'], optimizer=sgd2)
    autoencoder3.compile(loss=config['AE_LOSS'], optimizer=sgd3)

    encoder1.compile(loss=config['AE_LOSS'], optimizer=sgd1)
    encoder2.compile(loss=config['AE_LOSS'], optimizer=sgd2)
    encoder3.compile(loss=config['AE_LOSS'], optimizer=sgd3)

    # Layer-wise training
    early_stop1 = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.0001,
        patience=10,
        mode="auto",
        restore_best_weights=True,
        start_from_epoch=0
    )
    autoencoder1.fit(X_train, X_train,
                     epochs=config['AE_EPOCH'], batch_size=config['AE_BATCH_SIZE'],
                     # validation_split=0.20,
                     shuffle=False,
                     callbacks=[early_stop1]
                     )

    first_layer_code = encoder1.predict(X_train)
    early_stop2 = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.0001,
        patience=10,
        mode="auto",
        restore_best_weights=True,
        start_from_epoch=0
    )
    autoencoder2.fit(first_layer_code, first_layer_code,
                     epochs=config['AE_EPOCH'], batch_size=config['AE_BATCH_SIZE'],
                     # validation_split=0.20,
                     shuffle=False,
                     callbacks=[early_stop2]
                     )

    second_layer_code = encoder2.predict(first_layer_code)
    early_stop3 = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.0001,
        patience=10,
        mode="auto",
        restore_best_weights=True,
        start_from_epoch=0
    )
    autoencoder3.fit(second_layer_code, second_layer_code,
                     epochs=config['AE_EPOCH'], batch_size=config['AE_BATCH_SIZE'],
                     # validation_split=0.20,
                     shuffle=False,
                     callbacks=[early_stop3]
                     )

    # Setting the Weights of the Deep Autoencoder which has Learned in Layer-wise Training
    # encoded1_da = Dense(hidden_size[0], activation=config['AE_ACTIVATION'], name='encode1')(input_img)
    # encoded2_da = Dense(hidden_size[1], activation=config['AE_ACTIVATION'], name='latent')(encoded1_da)
    # decoded2_da = Dense(hidden_size[0], activation=config['AE_ACTIVATION'], name='decode1')(encoded2_da)
    # decoded1_da = Dense(X_train.shape[1], activation=config['AE_ACTIVATION'], name='out')(decoded2_da)

    # deep_autoencoder = Model(inputs=input_img, outputs=decoded1_da)
    # sgd5 = opt_factory_ae.get_opt()
    # deep_autoencoder.compile(loss=config['AE_LOSS'], optimizer=sgd5)
    #
    # deep_autoencoder.get_layer('encode1').set_weights(autoencoder1.layers[1].get_weights())  # first dense layer
    # deep_autoencoder.get_layer('latent').set_weights(autoencoder2.layers[1].get_weights())  # first decoder
    # deep_autoencoder.get_layer('decode1').set_weights(autoencoder2.layers[2].get_weights())  # second decoder
    # deep_autoencoder.get_layer('out').set_weights(autoencoder1.layers[2].get_weights())

    encoded1_da = Dense(hidden_size[0], activation=config['AE_ACTIVATION'], name='encode1')(input_img)
    encoded2_da = Dense(hidden_size[1], activation=config['AE_ACTIVATION'], name='encode2')(encoded1_da)
    encoded3_da = Dense(hidden_size[2], activation=config['AE_ACTIVATION'], name='encode3')(encoded2_da)
    decoded1_da = Dense(X_train.shape[1], activation=config['AE_ACTIVATION'], name='out')(encoded3_da)

    deep_autoencoder = Model(inputs=input_img, outputs=decoded1_da)

    deep_autoencoder.get_layer('encode1').set_weights(autoencoder1.layers[1].get_weights())
    deep_autoencoder.get_layer('encode2').set_weights(autoencoder2.layers[1].get_weights())
    deep_autoencoder.get_layer('encode3').set_weights(autoencoder3.layers[1].get_weights())

    sgd4 = opt_factory_ae.get_opt()
    deep_autoencoder.compile(loss=config['AE_LOSS'], optimizer=sgd4)
    early_stop4 = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.0001,
        patience=10,
        mode="auto",
        restore_best_weights=True,
        start_from_epoch=0
    )
    deep_autoencoder.fit(X_train, X_train,
                         epochs=config['AE_EPOCH'], batch_size=config['AE_BATCH_SIZE'],
                         shuffle=False,
                         callbacks=[early_stop4])

    print('END OF AUTOENCODER LAYER-WISE TRAINING.')
    return deep_autoencoder


if __name__ == '__main__':
    dataset_name = 'KDD_CUP99'
    mode = 'LW'
    # mode = 'MAGNETO'
    x_train, y_train, x_test, y_test, train_cols, test_cols = data(dataset_name)
    if mode == 'MAGNETO':
        ae = Autoencoder(x_train)
    elif mode == 'LW':
        ae = AutoencoderLayerWise(x_train)

    save_path = f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}'
    x_train_reconstruct = ae.predict(x_train)
    x_test_reconstruct = ae.predict(x_test)
    reconstructed_train_df = pd.DataFrame(np.concatenate((x_train_reconstruct, y_train), axis=1),
                                          columns=train_cols)
    reconstructed_test_df = pd.DataFrame(np.concatenate((x_test_reconstruct, y_test), axis=1),
                                         columns=test_cols)
    reconstructed_train_df.to_csv(os.path.join(save_path, f'train_binary_AE_{mode}.csv'), index=False)
    reconstructed_test_df.to_csv(os.path.join(save_path, f'test_binary_AE_{mode}.csv'), index=False)
    print('Reconstructed Train Dataset Saved with Shape of:', x_train_reconstruct.shape, y_train.shape)
    print('Reconstructed Test Dataset Saved with Shape of:', x_test_reconstruct.shape, y_test.shape)

    ae.save(os.path.join(save_path, f'Autoencoder_{mode}.h5'))
    print('AE saved in:', os.path.join(save_path, f'Autoencoder_{mode}.h5'))
