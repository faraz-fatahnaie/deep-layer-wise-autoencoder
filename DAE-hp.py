import argparse
import csv
import time
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.keras
from hyperopt import STATUS_OK, SparkTrials, hp, Trials, fmin, tpe
from tensorflow import expand_dims

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.python.client import device_lib

from sklearn.model_selection import train_test_split
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
# gpus = tf.config.list_physical_devices('GPU')
# device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
# print(device)
# if device == '/GPU:0':
#     # Set TensorFlow to use GPU
#     try:
#         # Restrict TensorFlow to only allocate memory as needed
#         tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
#
#         # Set the device to GPU
#         tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
#         print("Using GPU for computations.")
#     except RuntimeError as e:
#         print(e)
#         print("Error occurred. Using CPU instead.")
# else:
#     print("No GPU available. Using CPU for computations.")

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

tid = 0
best_loss = float('inf')
best_val_acc = 0
best_ae = None
best_params = dict()


def DAE(params_ae, method: str = 'layer-wise'):
    tf.keras.backend.clear_session()
    print(params_ae)

    time_file = open(Path(SAVE_PATH_).joinpath('time.txt'), 'w')
    train_time = 0
    tr_loss = 0
    val_loss = 0

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
        deep_autoencoder = tf.keras.models.load_model(ae_path)
        print(f'DAE loaded from {ae_path}')
        try:
            tr_loss = deep_autoencoder.evaluate(X_train, X_train)
            val_loss = deep_autoencoder.evaluate(X_val, X_val)
            print(f'loss: {tr_loss}, val_loss: {val_loss}')
        except:
            pass
    else:
        if method == 'formal':
            input_img = tensorflow.keras.Input(shape=(X_train.shape[1],))
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

            deep_autoencoder = tf.keras.models.Model(inputs=input_img, outputs=decoded1_da)

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
                # min_delta=0.0001,
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

        else:
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

            if method == 'layer-wise-encoder':
                autoencoder3, encoder3 = partial_ae_factory(in_shape=hidden_size[1],
                                                            hidden_size=X_train.shape[1],
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
                # min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )
            early_stop2 = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                # min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )
            early_stop_last = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                # min_delta=0.0001,
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
                                       callbacks=[early_stop_last],
                                       verbose=2
                                       )

            pae_train_end_time = time.time()
            pae_train_elapsed_time = int(pae_train_end_time - pae_train_start_time)

            input_img = tensorflow.keras.Input(shape=(X_train.shape[1],))
            encoded1_da = Dense(hidden_size[0], activation=params_ae['ae_activation'], name='encode1')(input_img)
            encoded2_da = Dense(hidden_size[1], activation=params_ae['ae_activation'], name='encode2')(encoded1_da)
            encoded3_da = Dense(hidden_size[2], activation=params_ae['ae_activation'], name='encode3')(encoded2_da)

            deep_autoencoder = tf.keras.models.Model(inputs=input_img, outputs=encoded3_da)

            deep_autoencoder.get_layer('encode1').set_weights(autoencoder1.layers[1].get_weights())
            deep_autoencoder.get_layer('encode2').set_weights(autoencoder2.layers[1].get_weights())
            deep_autoencoder.get_layer('encode3').set_weights(autoencoder3.layers[1].get_weights())

            outLayer_pae_elapsed_time = 0
            if method == 'layer-wise':
                decoded1_da = Dense(X_train.shape[1], activation=params_ae['ae_out_activation'], name='out',
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                    bias_initializer=tf.keras.initializers.Zeros()
                                    )(encoded3_da)
                deep_autoencoder = tf.keras.models.Model(inputs=input_img, outputs=decoded1_da)

                deep_autoencoder.get_layer('encode1').set_weights(autoencoder1.layers[1].get_weights())
                deep_autoencoder.get_layer('encode2').set_weights(autoencoder2.layers[1].get_weights())
                deep_autoencoder.get_layer('encode3').set_weights(autoencoder3.layers[1].get_weights())

                deep_autoencoder.get_layer('encode1').trainable = False
                deep_autoencoder.get_layer('encode2').trainable = False
                deep_autoencoder.get_layer('encode3').trainable = False

                sgd4 = opt_factory_ae.get_opt()
                deep_autoencoder.compile(loss=params_ae['ae_loss'], optimizer=sgd4)

                early_stop_last = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    # min_delta=0.0001,
                    patience=10,
                    mode="auto",
                    restore_best_weights=True
                )

                print('========== LAYER 4 ==========')
                outLayer_pae_start_time = time.time()
                history = deep_autoencoder.fit(X_train, X_train,
                                               validation_data=(X_val, X_val),
                                               epochs=epoch,
                                               batch_size=params_ae['ae_batch'],
                                               shuffle=False,
                                               callbacks=[early_stop_last],
                                               verbose=2
                                               )
                outLayer_pae_elapsed_time = time.time() - outLayer_pae_start_time

            train_time = int(pae_train_elapsed_time + outLayer_pae_elapsed_time)
            time_file.write(f'Autoencoder training (sec): {train_time}\n')

            stopped_epoch = early_stop_last.stopped_epoch

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


def train_cf(x_train, y_train, x_val, y_val, params):
    global tid
    global best_ae

    tid += 1
    tf.keras.backend.clear_session()
    print(params)
    x_train = np.array(x_train)
    x_val = np.array(x_val)

    if config['AE_FINETUNE']:
        n_features = x_train.shape[1]
    else:
        n_features = x_train.shape[2]

    if config['MODEL_NAME'] == 'BILSTM':
        cf = tf.keras.models.Sequential()
        cf.add(tensorflow.keras.Input(shape=(1, n_features)))

        forward_layer1 = LSTM(units=params['unit1'], return_sequences=True,
                              kernel_initializer='glorot_uniform', bias_initializer='zeros')
        backward_layer1 = LSTM(units=params['unit1'], return_sequences=True, go_backwards=True,
                               kernel_initializer='glorot_uniform', bias_initializer='zeros')
        cf.add(Bidirectional(forward_layer1, backward_layer=backward_layer1, merge_mode=params['merge_mode1']))

        cf.add(Dropout(params['dropout1']))

        forward_layer2 = LSTM(units=params['unit2'], return_sequences=True,
                              kernel_initializer='glorot_uniform', bias_initializer='zeros')
        backward_layer2 = LSTM(units=params['unit2'], return_sequences=True, go_backwards=True,
                               kernel_initializer='glorot_uniform', bias_initializer='zeros')
        cf.add(Bidirectional(forward_layer2, backward_layer=backward_layer2, merge_mode=params['merge_mode2']))

        cf.add(Dropout(params['dropout2']))

        cf.add(Flatten())
        cf.add(Dense(y_train.shape[1],
                     activation="softmax",
                     # kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                     # bias_regularizer=tf.keras.regularizers.L2(1e-4),
                     # activity_regularizer=tf.keras.regularizers.L2(1e-5)
                     ))

    elif config['MODEL_NAME'] == 'LSTM':
        cf = tf.keras.models.Sequential()
        cf.add(tensorflow.keras.Input(shape=(1, n_features)))

        cf.add(LSTM(params['unit1'], return_sequences=True))

        cf.add(Dropout(params['dropout1']))

        cf.add(LSTM(params['unit2'], return_sequences=True))

        cf.add(Dropout(params['dropout2']))

        cf.add(Flatten())
        cf.add(Dense(y_train.shape[1],
                     activation="softmax",
                     # kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                     # bias_regularizer=tf.keras.regularizers.L2(1e-4),
                     # activity_regularizer=tf.keras.regularizers.L2(1e-5)
                     ))

    if config['AE_FINETUNE']:
        input_img = tf.keras.Input(shape=(x_train.shape[1],))
        ae_out = best_ae(input_img)
        ae_out = expand_dims(ae_out, axis=1)
        cf_out = cf(ae_out)
        model = tf.keras.models.Model(inputs=input_img, outputs=cf_out)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(params["learning_rate"]),
                      metrics=['acc'])
    else:
        model = cf

    cf.compile(loss='categorical_crossentropy',
               optimizer=Adam(params["learning_rate"]),
               metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   min_delta=0.0001,
                                   patience=10,
                                   restore_best_weights=True),
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(CHECKPOINT_PATH_, 'best_model.h5'),
                                       monitor='val_loss',
                                       save_best_only=True)

    # best_acc_value = 1.0
    # if config['DATASET_NAME'] == 'KDD_CUP99':
    #     best_acc_value = None
    # elif config['DATASET_NAME'] == 'UNSW_NB15':
    #     best_acc_value = 0.9
    # elif config['DATASET_NAME'] == 'CICIDS':
    #     best_acc_value = 0.98
    # early_stopping2 = CustomEarlyStopping(best_max_value=best_acc_value)
    early_stopping2 = CustomEarlyStopping(best_max_value=1.0)

    train_start_time = time.time()
    model.fit(
        x_train,
        y_train,
        epochs=config["EPOCH"],
        verbose=2,
        validation_data=(x_val, y_val),
        batch_size=params["batch"],
        workers=4,
        callbacks=[early_stopping, early_stopping2, model_checkpoint]
    )
    train_end_time = time.time()

    model.load_weights(os.path.join(CHECKPOINT_PATH_, 'best_model.h5'))
    Y_predicted = model.predict(x_val, workers=4, verbose=2)

    y_val = np.argmax(y_val, axis=1)
    Y_predicted = np.argmax(Y_predicted, axis=1)

    cf = confusion_matrix(y_val, Y_predicted)
    acc = accuracy_score(y_val, Y_predicted)
    precision = precision_score(y_val, Y_predicted, average='binary')
    recall = recall_score(y_val, Y_predicted, average='binary')
    f1 = f1_score(y_val, Y_predicted, average='binary')
    epochs = early_stopping2.stopped_epoch + 1

    del x_train, x_val, y_train, y_val, Y_predicted

    return model, {
        "tid": tid,
        "epochs": epochs,
        "train_time": int(train_end_time - train_start_time),
        "unit1": params["unit1"],
        "unit2": params["unit2"],
        # "merge_mode1": params['merge_mode1'],
        # "merge_mode2": params['merge_mode2'],
        "learning_rate": params["learning_rate"],
        "batch": params["batch"],
        "dropout1": params["dropout1"],
        "dropout2": params["dropout2"],
        "TP_val": cf[0][0],
        "FP_val": cf[0][1],
        "TN_val": cf[1][1],
        "FN_val": cf[1][0],
        "OA_val": acc,
        "P_val": precision,
        "R_val": recall,
        "F1_val": f1,
    }


def hyperopt_cf(params):
    global SavedParameters
    global best_val_acc
    global best_ae

    print("start training")
    model, val = train_cf(XGlobal, YGlobal, XValGlobal, YValGlobal, params)

    print("start predicting")
    test_start_time = time.time()
    y_predicted = model.predict(XTestGlobal, workers=4, verbose=2)
    test_elapsed_time = time.time() - test_start_time

    y_predicted = np.argmax(y_predicted, axis=1)
    YTestGlobal_temp = np.argmax(YTestGlobal, axis=1)
    cm = confusion_matrix(YTestGlobal_temp, y_predicted)

    tf.keras.backend.clear_session()

    SavedParameters.append(val)

    SavedParameters[-1].update({
        "test_time": int(test_elapsed_time),
        "TP_test": cm[0][0],
        "FP_test": cm[0][1],
        "FN_test": cm[1][0],
        "TN_test": cm[1][1],
        "OA_test": metrics.accuracy_score(YTestGlobal_temp, y_predicted),
        "P_test": metrics.precision_score(YTestGlobal_temp, y_predicted, average='binary'),
        "R_test": metrics.recall_score(YTestGlobal_temp, y_predicted, average='binary'),
        "F1_test": metrics.f1_score(YTestGlobal_temp, y_predicted, average='binary'),
    })

    # Save model
    if SavedParameters[-1]["F1_val"] > best_val_acc:
        print("new model saved:" + str(SavedParameters[-1]))
        model.save(os.path.join(SAVE_PATH_, Name.replace(".csv", "_model.h5")))
        del model
        best_val_acc = SavedParameters[-1]["F1_val"]

    SavedParameters = sorted(SavedParameters, key=lambda i: i['F1_val'], reverse=True)

    try:
        with open((os.path.join(SAVE_PATH_, 'best_result.csv')), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")

    gc.collect()
    return {'loss': -val["F1_val"], 'status': STATUS_OK}


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
    # return {'loss': val["loss"], 'status': STATUS_OK}


def train_DAE(dataset_name):
    global YGlobal
    global YValGlobal
    global YTestGlobal
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
    project = 'DAE'
    config = {}
    BASE_DIR = Path(__file__).resolve().parent
    while flag:

        config_name = f'CONFIG_{dataset_name}'
        config_dir = BASE_DIR.joinpath('configs')
        config_file = open(f'{config_dir}/{config_name}.json')
        config_file = json.load(config_file)
        config = setting_DAE(config_file=config_file, project=project)
        TEMP_FILENAME = f"{config['DATASET_NAME']}-{config['CLASSIFICATION_MODE']}-{config['MODEL_NAME']}-{i}"
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
    base_path = Path(__file__).resolve().parent
    train = pd.read_csv(base_path.joinpath(f'dataset\\{dataset_name}\\train_binary.csv'))
    test = pd.read_csv(base_path.joinpath(f'dataset\\{dataset_name}\\test_binary.csv'))

    XGlobal, YGlobal = parse_data(train, config['DATASET_NAME'], config['CLASSIFICATION_MODE'])
    XTestGlobal, YTestGlobal = parse_data(test, config['DATASET_NAME'], config['CLASSIFICATION_MODE'])

    YGlobal = tf.keras.utils.to_categorical(YGlobal, num_classes=2)
    YTestGlobal = tf.keras.utils.to_categorical(YTestGlobal, num_classes=2)

    XGlobal, XValGlobal, YGlobal, YValGlobal = train_test_split(XGlobal,
                                                                YGlobal,
                                                                test_size=config['VAL_SIZE'],
                                                                stratify=YGlobal,
                                                                random_state=config['SEED']
                                                                )

    print('train set:', XGlobal.shape, YGlobal.shape)
    print('validation set:', XValGlobal.shape, YValGlobal.shape)
    print('test set:', XTestGlobal.shape, YTestGlobal.shape)

    # XGlobal = tf.convert_to_tensor(XGlobal)
    # XTestGlobal = tf.convert_to_tensor(XTestGlobal)
    # XValGlobal = tf.convert_to_tensor(XValGlobal)
    # YGlobal = tf.convert_to_tensor(YGlobal)
    # YTestGlobal = tf.convert_to_tensor(YTestGlobal)
    # YValGlobal = tf.convert_to_tensor(YValGlobal)
    #
    # if device == '':
    #     print("Moving data to GPU...")
    #     XGlobal = XGlobal.gpu()
    #     XValGlobal = XValGlobal.gpu()
    #     XTestGlobal = XTestGlobal.gpu()
    #     print("Data moved to GPU.")

    ae_hyperparameters_to_optimize = {
        "ae_activation": config['AE_ACTIVATION'],
        "ae_out_activation": config['AE_O_ACTIVATION'],
        "ae_loss": config['AE_LOSS'],
        "ae_optimizer": config['AE_OPTIMIZER']
    }
    keys = list(ae_hyperparameters_to_optimize.keys())
    values = list(ae_hyperparameters_to_optimize.values())
    for combination in product(*values):
        params_ae = {keys[i]: combination[i] for i in range(len(keys))}
        params_ae['ae_unit'] = 128
        params_ae['ae_batch'] = 32
        hyperopt_ae(params_ae, config['AE_METHOD'])

    print(best_params)

    # best_ae = keras.models.load_model('C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\'
    #                                   'trained_ae\\'
    #                                   'AE_KDD_CUP99_128_128_128_A-tanh_OA-linear_mse_E150_B32_sgd_layer-wise.keras')

    if not config['AE_FINETUNE']:
        XGlobal = best_ae.predict(XGlobal, verbose=2)
        XValGlobal = best_ae.predict(XValGlobal, verbose=2)
        XTestGlobal = best_ae.predict(XTestGlobal, verbose=2)
        XGlobal = np.reshape(XGlobal, (-1, 1, XGlobal.shape[1]))
        XValGlobal = np.reshape(XValGlobal, (-1, 1, XValGlobal.shape[1]))
        XTestGlobal = np.reshape(XTestGlobal, (-1, 1, XTestGlobal.shape[1]))

    print('train set:', XGlobal.shape, YGlobal.shape)
    print('validation set:', XValGlobal.shape, YValGlobal.shape)
    print('test set:', XTestGlobal.shape, YTestGlobal.shape)
    tf.keras.backend.clear_session()

    cf_hyperparameters = {
        "unit1": hp.choice("unit1", config['UNIT']),
        "unit2": hp.choice("unit2", config['UNIT']),
        "batch": hp.choice("batch", config['BATCH']),
        # "epoch": hp.choice("epoch", config['EPOCH']),
        'dropout1': hp.uniform("dropout1", config['MIN_DROPOUT'], config['MAX_DROPOUT']),
        'dropout2': hp.uniform("dropout2", config['MIN_DROPOUT'], config['MAX_DROPOUT']),
        "learning_rate": hp.uniform("learning_rate", config['MIN_LR'], config['MAX_LR'])
    }
    if config['MODEL_NAME'] == 'BILSTM':
        cf_hyperparameters['merge_mode2'] = hp.choice("merge_mode2", config['MERGE_MODE'])
        cf_hyperparameters['merge_mode2'] = hp.choice("merge_mode2", config['MERGE_MODE'])

    trials = Trials()
    # spark_trials = SparkTrials()
    fmin(hyperopt_cf, cf_hyperparameters,
         trials=trials,
         algo=tpe.suggest,
         max_evals=config['MAX_EVALS'],
         rstate=np.random.default_rng(config['SEED']))
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--dataset', type=str, default='UNSW_NB15',
                        help='dataset name choose from: "UNSW", "KDD", "CICIDS"')

    args = parser.parse_args()
    train_DAE(args.dataset)
