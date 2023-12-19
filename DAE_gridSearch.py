import argparse
import csv
import time
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import STATUS_OK, SparkTrials, hp, Trials, fmin, tpe
from keras import backend as K
from keras import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Flatten, Concatenate
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.python.client import device_lib
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model
from utils import parse_data, result, OptimizerFactory, set_seed, CustomEarlyStopping
from layer_wise_autoencoder import partial_ae_factory
import os
from pathlib import Path
from configs.setting_DAE import setting_DAE
import json
import gc

# tf.compat.v1.disable_eager_execution()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(device_lib.list_local_devices())

config = dict()

XGlobal = list()
YGlobal = list()

XValGlobal = list()
YValGlobal = list()

XTestGlobal = list()
YTestGlobal = list()

SavedParameters = list()
SavedParametersAE = list()
Mode = str()
Name = str()
SAVE_PATH_ = str()
CHECKPOINT_PATH_ = str()

best_ae = None
best_params = dict()

tid = 0
best_loss = float('inf')
best_val_acc = 0


def DAE(params_ae, method: str = 'layer-wise'):

    K.clear_session()
    print(params_ae)
    time_file = open(Path(SAVE_PATH_).joinpath('time.txt'), 'w')
    train_time = 0

    dataset_name = config['DATASET_NAME']
    epoch = config['AE_EPOCH']
    hidden_size = [params_ae['ae_unit'], params_ae['ae_unit'], params_ae['ae_unit']]
    batch_size = params_ae['ae_batch']
    activation = params_ae['ae_activation']
    out_activation = params_ae['ae_out_activation']
    loss_fn = params_ae['ae_loss']
    opt = params_ae['ae_optimizer']

    ae_filename = f'AE_{dataset_name}_' + '_'.join(
        map(str,
            hidden_size)) + f'_A-{activation}_OA-{out_activation}_{loss_fn}_E{epoch}_B{batch_size}_{opt}_{method}'

    ae_path = Path(
        f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\trained_ae\\{ae_filename}.keras')

    X_train = np.array(XGlobal)
    X_val = np.array(XValGlobal)

    if os.path.isfile(ae_path) and not config['AE_TRAINABLE']:
        deep_autoencoder = keras.models.load_model(ae_path)
        print(f'DAE loaded from {ae_path}')
        try:
            tr_loss = deep_autoencoder.evaluate(X_train, X_train)
            val_loss = deep_autoencoder.evaluate(X_val, X_val)
            print(f'loss: {tr_loss}, val_loss: {val_loss}')
        except:
            pass
    else:
        if method == 'layer-wise':
            # ================= LAYER 1 =================
            autoencoder1, encoder1 = partial_ae_factory(in_shape=X_train.shape[1],
                                                        hidden_size=hidden_size[0],
                                                        activation=params_ae['ae_activation'])

            # ================= LAYER 2 =================
            autoencoder2, encoder2 = partial_ae_factory(in_shape=hidden_size[0],
                                                        hidden_size=hidden_size[1],
                                                        activation=params_ae['ae_activation'])
            # ================= LAYER 3 =================
            autoencoder3, encoder3 = partial_ae_factory(in_shape=hidden_size[1],
                                                        hidden_size=hidden_size[2],
                                                        activation=params_ae['ae_activation'])

            opt_factory_ae = OptimizerFactory(opt=params_ae['ae_optimizer'],
                                              lr_schedule=config['AE_SCHEDULE'],
                                              len_dataset=len(X_train),
                                              epochs=epoch,
                                              batch_size=params_ae['ae_batch'],
                                              init_lr=config['AE_INITIAL_LR'],
                                              final_lr=config['AE_FINAL_LR'])

            sgd1 = opt_factory_ae.get_opt()
            sgd2 = opt_factory_ae.get_opt()
            sgd3 = opt_factory_ae.get_opt()

            autoencoder1.compile(loss=params_ae['ae_loss'], optimizer=sgd1)
            autoencoder2.compile(loss=params_ae['ae_loss'], optimizer=sgd2)
            autoencoder3.compile(loss=params_ae['ae_loss'], optimizer=sgd3)

            encoder1.compile(loss=params_ae['ae_loss'], optimizer=sgd1)
            encoder2.compile(loss=params_ae['ae_loss'], optimizer=sgd2)
            encoder3.compile(loss=params_ae['ae_loss'], optimizer=sgd3)

            early_stop1 = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )
            early_stop2 = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )
            early_stop3 = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )

            print('========== LAYER 1 ==========')
            pae_train_start_time = time.time()
            autoencoder1.fit(X_train, X_train,
                             validation_data=(X_val, X_val),
                             epochs=epoch,
                             batch_size=params_ae['ae_batch'],
                             shuffle=False,
                             callbacks=[early_stop1],
                             verbose=2
                             )

            print('========== LAYER 2 ==========')
            first_layer_code = encoder1.predict(X_train, verbose=2)
            first_layer_code_val = encoder1.predict(X_val, verbose=2)
            autoencoder2.fit(first_layer_code, first_layer_code,
                             validation_data=(first_layer_code_val, first_layer_code_val),
                             epochs=epoch,
                             batch_size=params_ae['ae_batch'],
                             shuffle=False,
                             callbacks=[early_stop2],
                             verbose=2
                             )

            print('========== LAYER 3 ==========')
            second_layer_code = encoder2.predict(first_layer_code, verbose=2)
            second_layer_code_val = encoder2.predict(first_layer_code_val, verbose=2)
            history = autoencoder3.fit(second_layer_code, second_layer_code,
                                       validation_data=(second_layer_code_val, second_layer_code_val),
                                       epochs=epoch,
                                       batch_size=params_ae['ae_batch'],
                                       shuffle=False,
                                       callbacks=[early_stop3],
                                       verbose=2
                                       )

            pae_train_end_time = time.time()
            pae_train_elapsed_time = int(pae_train_end_time - pae_train_start_time)

            input_img = Input(shape=(X_train.shape[1],))
            encoded1_da = Dense(hidden_size[0], activation=params_ae['ae_activation'], name='encode1')(input_img)
            encoded2_da = Dense(hidden_size[1], activation=params_ae['ae_activation'], name='encode2')(encoded1_da)
            encoded3_da = Dense(hidden_size[2], activation=params_ae['ae_activation'], name='encode3')(encoded2_da)
            decoded1_da = Dense(X_train.shape[1], activation=params_ae['ae_out_activation'], name='out',
                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                bias_initializer=tf.keras.initializers.Zeros()
                                )(encoded3_da)

            deep_autoencoder = Model(inputs=input_img, outputs=decoded1_da)

            deep_autoencoder.get_layer('encode1').set_weights(autoencoder1.layers[1].get_weights())
            deep_autoencoder.get_layer('encode2').set_weights(autoencoder2.layers[1].get_weights())
            deep_autoencoder.get_layer('encode3').set_weights(autoencoder3.layers[1].get_weights())

            # deep_autoencoder.get_layer('encode1').trainable = False
            # deep_autoencoder.get_layer('encode2').trainable = False
            # deep_autoencoder.get_layer('encode3').trainable = False

            sgd4 = opt_factory_ae.get_opt()
            deep_autoencoder.compile(loss=params_ae['ae_loss'], optimizer=sgd4)

            early_stop4 = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )

            print('========== LAYER 4 ==========')
            outLayer_pae_start_time = time.time()
            history = deep_autoencoder.fit(X_train, X_train,
                                           validation_data=(X_val, X_val),
                                           epochs=epoch, batch_size=params_ae['ae_batch'],
                                           shuffle=False,
                                           callbacks=[early_stop4],
                                           verbose=2
                                           )
            outLayer_pae_elapsed_time = time.time() - outLayer_pae_start_time

            train_time = int(pae_train_elapsed_time + outLayer_pae_elapsed_time)
            time_file.write(f'Autoencoder training (sec): {train_time}\n')

            stopped_epoch = early_stop4.stopped_epoch

            tr_loss = history.history['loss'][stopped_epoch]
            val_loss = history.history['val_loss'][stopped_epoch]

            deep_autoencoder.save(Path(SAVE_PATH_).joinpath("DAE.keras"))
            deep_autoencoder.save(ae_path)

        else:

            input_img = Input(shape=(X_train.shape[1],))
            encoded1_da = Dense(hidden_size[0], activation=params_ae['ae_activation'], name='encode1',
                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                bias_initializer=tf.keras.initializers.Zeros()
                                )(input_img)
            encoded2_da = Dense(hidden_size[1], activation=params_ae['ae_activation'], name='encode2',
                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                bias_initializer=tf.keras.initializers.Zeros()
                                )(encoded1_da)
            encoded3_da = Dense(hidden_size[2], activation=params_ae['ae_activation'], name='encode3',
                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                bias_initializer=tf.keras.initializers.Zeros()
                                )(encoded2_da)
            decoded1_da = Dense(X_train.shape[1], activation=params_ae['ae_out_activation'], name='out',
                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                bias_initializer=tf.keras.initializers.Zeros()
                                )(encoded3_da)

            deep_autoencoder = Model(inputs=input_img, outputs=decoded1_da)

            opt_factory_ae = OptimizerFactory(opt=params_ae['ae_optimizer'],
                                              lr_schedule=config['AE_SCHEDULE'],
                                              len_dataset=len(X_train),
                                              epochs=epoch,
                                              batch_size=params_ae['ae_batch'],
                                              init_lr=config['AE_INITIAL_LR'],
                                              final_lr=config['AE_FINAL_LR'])

            sgd = opt_factory_ae.get_opt()
            deep_autoencoder.compile(loss=params_ae['ae_loss'], optimizer=sgd)

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )

            ae_start_time = time.time()
            history = deep_autoencoder.fit(X_train, X_train,
                                           validation_data=(X_val, X_val),
                                           epochs=config['AE_EPOCH'], batch_size=params_ae['ae_batch'],
                                           shuffle=False,
                                           callbacks=[early_stop],
                                           verbose=2
                                           )
            ae_elapsed_time = time.time() - ae_start_time

            train_time = int(ae_elapsed_time)
            time_file.write(f'Autoencoder training (sec): {train_time}\n')

            stopped_epoch = early_stop.stopped_epoch

            tr_loss = history.history['loss'][stopped_epoch]
            val_loss = history.history['val_loss'][stopped_epoch]

            deep_autoencoder.save(Path(SAVE_PATH_).joinpath("DAE.keras"))
            deep_autoencoder.save(ae_path)

    return deep_autoencoder, {"loss": tr_loss,
                              "val_loss": val_loss,
                              "ae_loss": params_ae['ae_loss'],
                              "ae_optimizer": params_ae['ae_optimizer'],
                              "ae_activation": params_ae['ae_activation'],
                              "ae_out_activation": params_ae['ae_out_activation'],
                              "ae_batch": params_ae['ae_batch'],
                              "ae_unit": params_ae['ae_unit'],
                              "train_time": train_time
                              }


def hyperopt_ae(params_ae, method):
    global SavedParametersAE
    global best_loss
    global best_ae
    global best_params

    ae, param = DAE(params_ae, method)

    SavedParametersAE.append(param)
    # Save model
    if SavedParametersAE[-1]["val_loss"] < best_loss:
        print("new saved model:" + str(SavedParametersAE[-1]))
        best_ae = ae
        best_params = param
        ae.save(os.path.join(SAVE_PATH_, Name.replace(".csv", "ae_model.h5")))
        del ae
        best_loss = SavedParametersAE[-1]["val_loss"]

    SavedParametersAE = sorted(SavedParametersAE, key=lambda i: i['val_loss'])

    try:
        with open((os.path.join(SAVE_PATH_, 'best_result_ae.csv')), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParametersAE[0].keys())
            writer.writeheader()
            writer.writerows(SavedParametersAE)
    except IOError:
        print("I/O error")
    gc.collect()


def train_DAE(dataset_name):
    global YGlobal
    global YTestGlobal
    global YValGlobal
    global XGlobal
    global XValGlobal
    global XTestGlobal

    global best_ae
    global best_params

    global config
    global SAVE_PATH_
    global CHECKPOINT_PATH_
    i = 1
    flag = True
    project = 'AE_training'
    config = {}
    BASE_DIR = Path(__file__).resolve().parent
    while flag:

        config_name = f'CONFIG_{dataset_name}'
        config_dir = BASE_DIR.joinpath('configs')
        config_file = open(f'{config_dir}/{config_name}.json')
        config_file = json.load(config_file)
        config = setting_DAE(config_file=config_file, project=project)
        TEMP_FILENAME = f"AE-{config['AE_METHOD']}-{config['DATASET_NAME']}-{i}"
        TEMP_PATH = BASE_DIR.joinpath(f"session_{project}/{TEMP_FILENAME}")

        if os.path.isdir(TEMP_PATH):
            i += 1
        else:
            flag = False

            os.mkdir(BASE_DIR.joinpath(f"session_{project}/{TEMP_FILENAME}"))
            SAVE_PATH_ = BASE_DIR.joinpath(f"session_{project}/{TEMP_FILENAME}")

            os.mkdir(BASE_DIR.joinpath(f'{SAVE_PATH_}/model_checkpoint'))
            CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f"model_checkpoint/")

            with open(f'{SAVE_PATH_}/MODEL_CONFIG.json', 'w') as f:
                json.dump(config_file, f)

            time_file = open(SAVE_PATH_.joinpath('time.txt'), 'w')
            time_file.write('Result Time \n')

            print(f'MODEL SESSION: {SAVE_PATH_}')

    if config['SEED'] is not None:
        set_seed(seed=config['SEED'])

    # Generate synthetic data for testing (replace with your dataset)
    dataset_name = config['DATASET_NAME']
    train = pd.read_csv(
        f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}'
        f'\\train_binary.csv')
    test = pd.read_csv(
        f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}'
        f'\\test_binary.csv')

    XGlobal, YGlobal = parse_data(train, config['DATASET_NAME'], config['CLASSIFICATION_MODE'])
    XTestGlobal, YTestGlobal = parse_data(test, config['DATASET_NAME'], config['CLASSIFICATION_MODE'])

    YGlobal = keras.utils.to_categorical(YGlobal, num_classes=2)
    YTestGlobal = keras.utils.to_categorical(YTestGlobal, num_classes=2)

    XGlobal, XValGlobal, YGlobal, YValGlobal = train_test_split(XGlobal,
                                                                YGlobal,
                                                                test_size=config['VAL_SIZE'],
                                                                stratify=YGlobal,
                                                                random_state=config['SEED']
                                                                )

    print('train set:', XGlobal.shape, YGlobal.shape)
    print('validation set:', XValGlobal.shape, YValGlobal.shape)
    print('test set:', XTestGlobal.shape, YTestGlobal.shape)

    ae_hyperparameters_to_optimize = {
        "ae_activation": ['tanh', 'relu', 'sigmoid'],
        "ae_out_activation": ['relu', 'linear', 'sigmoid'],
        "ae_loss": ['mse', 'mae'],
        "ae_optimizer": ['adam', 'sgd']
    }
    keys = list(ae_hyperparameters_to_optimize.keys())
    values = list(ae_hyperparameters_to_optimize.values())
    for combination in product(*values):
        params_ae = {keys[i]: combination[i] for i in range(len(keys))}
        params_ae['ae_unit'] = 128
        params_ae['ae_batch'] = 32
        hyperopt_ae(params_ae, config['AE_METHOD'])

    print(best_params)
    # ae_hyperparameters_to_optimize = {
    #     "ae_unit": [32, 64, 128],
    #     "ae_batch": [32, 64, 128]
    # }
    # keys = list(ae_hyperparameters_to_optimize.keys())
    # values = list(ae_hyperparameters_to_optimize.values())
    # for combination in product(*values):
    #     params_ae = {keys[i]: combination[i] for i in range(len(keys))}
    #     params_ae['ae_activation'] = str(best_params['ae_activation'])
    #     params_ae['ae_out_activation'] = str(best_params['ae_out_activation'])
    #     params_ae['ae_loss'] = str(best_params['ae_loss'])
    #     params_ae['ae_optimizer'] = str(best_params['ae_optimizer'])
    #     hyperopt_ae(params_ae, config['AE_METHOD'])
    # print('BEST AE PARAMS:', best_params)

    # best_ae = keras.models.load_model('C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\'
    #                                   'trained_ae_temp2\\AE_UNSW_NB15_32_32_32_tanh_mae_E150_B32.keras')

    XGlobal = best_ae.predict(XGlobal, verbose=2)
    XTestGlobal = best_ae.predict(XTestGlobal, verbose=2)
    XGlobal = np.reshape(XGlobal, (-1, 1, XGlobal.shape[1]))
    XTestGlobal = np.reshape(XTestGlobal, (-1, 1, XTestGlobal.shape[1]))
    print(np.shape(XGlobal), np.shape(YGlobal), np.shape(XTestGlobal), np.shape(YTestGlobal))
    K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--dataset', type=str, default='UNSW_NB15',
                        help='dataset name choose from: "UNSW", "KDD", "CICIDS"')

    args = parser.parse_args()
    train_DAE(args.dataset)
