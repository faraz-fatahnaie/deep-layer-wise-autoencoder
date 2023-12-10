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
from configs.setting import setting
import json
import gc

# tf.compat.v1.disable_eager_execution()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(device_lib.list_local_devices())

config = dict()
XGlobal = list()
YGlobal = list()

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


def DAE(params_ae):
    K.clear_session()
    print(params_ae)
    time_file = open(Path(SAVE_PATH_).joinpath('time.txt'), 'w')
    train_time = 0

    dataset_name = config['DATASET_NAME']
    # h1, h2 = params_ae['ae_unit']
    # hidden_size = [h1, h2, h1]
    hidden_size = [params_ae['ae_unit'], params_ae['ae_unit'], params_ae['ae_unit'], params_ae['ae_unit']]
    activation = config['AE_ACTIVATION']
    loss_fn = config['AE_LOSS']
    ae_epoch = config['AE_EPOCH']
    ae_bs = params_ae['ae_batch']
    ae_filename = f'AE_{dataset_name}_' + '_'.join(
        map(str, hidden_size)) + f'_{activation}_{loss_fn}_E{ae_epoch}_B{ae_bs}'
    ae_path = Path(
        f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\trained_ae\\{ae_filename}.keras')

    X_train = np.array(XGlobal)
    if os.path.isfile(ae_path) and not config['AE_TRAINABLE']:
        deep_autoencoder = keras.models.load_model(ae_path)
        print(f'DAE loaded from {ae_path}')
        try:
            loss = deep_autoencoder.evaluate(X_train, X_train)
            print(loss)
        except:
            pass
    else:
        # ================= LAYER 1 =================
        autoencoder1, encoder1 = partial_ae_factory(in_shape=X_train.shape[1],
                                                    hidden_size=hidden_size[0],
                                                    activation=config['AE_ACTIVATION'])

        # ================= LAYER 2 =================
        autoencoder2, encoder2 = partial_ae_factory(in_shape=hidden_size[0],
                                                    hidden_size=hidden_size[1],
                                                    activation=config['AE_ACTIVATION'])
        # ================= LAYER 3 =================
        autoencoder3, encoder3 = partial_ae_factory(in_shape=hidden_size[1],
                                                    hidden_size=hidden_size[2],
                                                    activation=config['AE_ACTIVATION'])
        # # ================= LAYER 4 =================
        # autoencoder4, encoder4 = partial_ae_factory(in_shape=hidden_size[2],
        #                                             hidden_size=hidden_size[3],
        #                                             activation=config['AE_ACTIVATION'])

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
        # sgd4 = opt_factory_ae.get_opt()

        autoencoder1.compile(loss=config['AE_LOSS'], optimizer=sgd1)
        autoencoder2.compile(loss=config['AE_LOSS'], optimizer=sgd2)
        autoencoder3.compile(loss=config['AE_LOSS'], optimizer=sgd3)
        # autoencoder4.compile(loss=config['AE_LOSS'], optimizer=sgd4)

        encoder1.compile(loss=config['AE_LOSS'], optimizer=sgd1)
        encoder2.compile(loss=config['AE_LOSS'], optimizer=sgd2)
        encoder3.compile(loss=config['AE_LOSS'], optimizer=sgd3)
        # encoder4.compile(loss=config['AE_LOSS'], optimizer=sgd4)

        early_stop1 = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.0001,
            patience=10,
            mode="auto",
            restore_best_weights=True,
            start_from_epoch=0
        )
        early_stop2 = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.0001,
            patience=10,
            mode="auto",
            restore_best_weights=True,
            start_from_epoch=0
        )
        early_stop3 = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.0001,
            patience=10,
            mode="auto",
            restore_best_weights=True,
            start_from_epoch=0
        )
        early_stop4 = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.0001,
            patience=10,
            mode="auto",
            restore_best_weights=True,
            start_from_epoch=0
        )

        print('========== LAYER 1 ==========')
        pae_train_start_time = time.time()
        autoencoder1.fit(X_train, X_train,
                         epochs=config['AE_EPOCH'], batch_size=params_ae['ae_batch'],
                         shuffle=False,
                         callbacks=[early_stop1],
                         verbose=2
                         )

        print('========== LAYER 2 ==========')
        first_layer_code = encoder1.predict(X_train, verbose=2)
        autoencoder2.fit(first_layer_code, first_layer_code,
                         epochs=config['AE_EPOCH'], batch_size=params_ae['ae_batch'],
                         shuffle=False,
                         callbacks=[early_stop2],
                         verbose=2
                         )

        print('========== LAYER 3 ==========')
        second_layer_code = encoder2.predict(first_layer_code, verbose=2)
        history = autoencoder3.fit(second_layer_code, second_layer_code,
                                   epochs=config['AE_EPOCH'], batch_size=params_ae['ae_batch'],
                                   shuffle=False,
                                   callbacks=[early_stop3],
                                   verbose=2
                                   )

        # print('========== LAYER 3 ==========')
        # third_layer_code = encoder3.predict(second_layer_code, verbose=2)
        # history = autoencoder4.fit(third_layer_code, third_layer_code,
        #                            epochs=config['AE_EPOCH'], batch_size=params_ae['ae_batch'],
        #                            shuffle=False,
        #                            callbacks=[early_stop4],
        #                            verbose=2
        #                            )
        pae_train_end_time = time.time()
        pae_train_elapsed_time = int(pae_train_end_time - pae_train_start_time)

        input_img = Input(shape=(X_train.shape[1],))
        encoded1_da = Dense(hidden_size[0], activation=config['AE_ACTIVATION'], name='encode1')(input_img)
        encoded2_da = Dense(hidden_size[1], activation=config['AE_ACTIVATION'], name='encode2')(encoded1_da)
        encoded3_da = Dense(hidden_size[2], activation=config['AE_ACTIVATION'], name='encode3')(encoded2_da)
        # encoded4_da = Dense(hidden_size[3], activation=config['AE_ACTIVATION'], name='encode4')(encoded3_da)
        decoded1_da = Dense(X_train.shape[1], activation=config['AE_ACTIVATION'], name='out')(encoded3_da)

        deep_autoencoder = Model(inputs=input_img, outputs=decoded1_da)

        deep_autoencoder.get_layer('encode1').set_weights(autoencoder1.layers[1].get_weights())
        deep_autoencoder.get_layer('encode2').set_weights(autoencoder2.layers[1].get_weights())
        deep_autoencoder.get_layer('encode3').set_weights(autoencoder3.layers[1].get_weights())
        # deep_autoencoder.get_layer('encode4').set_weights(autoencoder4.layers[1].get_weights())

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

        print('========== LAYER 4 ==========')
        outLayer_pae_start_time = time.time()
        history = deep_autoencoder.fit(X_train, X_train,
                                       epochs=config['AE_EPOCH'], batch_size=params_ae['ae_batch'],
                                       shuffle=False,
                                       callbacks=[early_stop4],
                                       verbose=2
                                       )
        outLayer_pae_elapsed_time = time.time() - outLayer_pae_start_time

        train_time = int(pae_train_elapsed_time + outLayer_pae_elapsed_time)
        time_file.write(f'Autoencoder training (sec): {train_time}\n')

        loss = history.history['loss'][-1]
        deep_autoencoder.save(Path(SAVE_PATH_).joinpath("DAE.keras"))
        deep_autoencoder.save(ae_path)

    return deep_autoencoder, {"loss": loss,
                              "ae_batch": params_ae['ae_batch'],
                              "ae_unit": params_ae['ae_unit'],
                              "train_time": train_time
                              }


def train_cf(X, y, params):
    global tid
    tid += 1
    K.clear_session()
    print(params)
    x_train, x_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size=0.1,
                                                      stratify=y,
                                                      random_state=config['SEED']
                                                      )
    x_train = np.array(x_train)
    x_val = np.array(x_val)

    model = keras.models.Sequential()
    model.add(Input(shape=(1, x_train.shape[2])))

    forward_layer = LSTM(units=params['unit'], return_sequences=True)
    backward_layer = LSTM(units=params['unit'], return_sequences=True, go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode=params['merge_mode']))

    model.add(Dropout(params['dropout']))

    model.add(Flatten())
    model.add(Dense(y_train.shape[1],
                    activation="softmax",
                    # kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                    # bias_regularizer=tf.keras.regularizers.L2(1e-4),
                    # activity_regularizer=tf.keras.regularizers.L2(1e-5)
                    ))



    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(params["learning_rate"]),
                  metrics=['acc'])

    early_stopping1 = EarlyStopping(monitor='val_loss',
                                    mode='min',
                                    min_delta=0.0001,
                                    patience=10,
                                    restore_best_weights=True),
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(CHECKPOINT_PATH_, 'best_model.h5'),
                                       monitor='val_loss',
                                       save_best_only=True)

    best_acc_value = 1.0
    if config['DATASET_NAME'] == 'KDD_CUP99':
        best_acc_value = None
    elif config['DATASET_NAME'] == 'UNSW_NB15':
        best_acc_value = 0.9
    elif config['DATASET_NAME'] == 'CICIDS':
        best_acc_value = 0.98
    early_stopping2 = CustomEarlyStopping(best_max_value=best_acc_value)

    train_start_time = time.time()
    model.fit(
        x_train,
        y_train,
        epochs=config['EPOCHS'],
        verbose=2,
        validation_data=(x_val, y_val),
        batch_size=params["batch"],
        workers=4,
        callbacks=[early_stopping1, early_stopping2, model_checkpoint]
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

    del X, y, x_train, x_val, y_train, y_val, Y_predicted

    return model, {
        "tid": tid,
        "epochs": epochs,
        "train_time": int(train_end_time - train_start_time),
        "unit": params["unit"],
        "merge_mode": params['merge_mode'],
        "learning_rate": params["learning_rate"],
        "batch": params["batch"],
        "dropout": params["dropout"],
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
    model, val = train_cf(XGlobal, YGlobal, params)

    print("start predicting")
    test_start_time = time.time()
    y_predicted = model.predict(XTestGlobal, workers=4, verbose=2)
    test_elapsed_time = time.time() - test_start_time

    y_predicted = np.argmax(y_predicted, axis=1)
    YTestGlobal_temp = np.argmax(YTestGlobal, axis=1)
    cm = confusion_matrix(YTestGlobal_temp, y_predicted)

    K.clear_session()

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


def hyperopt_ae(params_ae):
    global SavedParametersAE
    global best_loss
    global best_ae

    ae, val = DAE(params_ae)

    SavedParametersAE.append(val)
    # Save model
    if SavedParametersAE[-1]["loss"] < best_loss:
        print("new saved model:" + str(SavedParametersAE[-1]))
        best_ae = ae
        ae.save(os.path.join(SAVE_PATH_, Name.replace(".csv", "ae_model.h5")))
        del ae
        best_loss = SavedParametersAE[-1]["loss"]

    SavedParametersAE = sorted(SavedParametersAE, key=lambda i: i['loss'])

    try:
        with open((os.path.join(SAVE_PATH_, 'best_result_ae.csv')), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParametersAE[0].keys())
            writer.writeheader()
            writer.writerows(SavedParametersAE)
    except IOError:
        print("I/O error")

    gc.collect()
    # return {'loss': val["loss"], 'status': STATUS_OK}


def train_DAE():
    global YGlobal
    global YTestGlobal
    global XGlobal
    global XTestGlobal

    global best_ae

    global config
    global SAVE_PATH_
    global CHECKPOINT_PATH_
    i = 1
    flag = True
    project = 'DAE'
    TRAINED_MODEL_PATH_ = ''
    config = {}
    BASE_DIR = Path(__file__).resolve().parent
    while flag:

        config, config_file = setting(project=project)
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
    print(XGlobal.shape, YGlobal.shape, XTestGlobal.shape, YTestGlobal.shape)

    ae_hyperparameters_to_optimize = {
        # "ae_unit": [(m, n, v) for m in [32, 64, 128] for n in [32, 64, 128] for v in [32, 64, 128] if m >= n >= v],
        # "ae_unit": [(m, n) for m in [32, 64, 128] for n in [32, 64, 128] if m >= n],
        "ae_unit": [32, 64, 128],
        "ae_batch": [32, 64, 128]}
    keys = list(ae_hyperparameters_to_optimize.keys())
    values = list(ae_hyperparameters_to_optimize.values())
    for combination in product(*values):
        params_ae = {keys[i]: combination[i] for i in range(len(keys))}
        hyperopt_ae(params_ae)

    # best_ae = keras.models.load_model('C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\'
    #                                   'trained_ae_temp2\\AE_UNSW_NB15_32_32_32_tanh_mae_E150_B32.keras')

    XGlobal = best_ae.predict(XGlobal, verbose=2)
    XTestGlobal = best_ae.predict(XTestGlobal, verbose=2)
    XGlobal = np.reshape(XGlobal, (-1, 1, XGlobal.shape[1]))
    XTestGlobal = np.reshape(XTestGlobal, (-1, 1, XTestGlobal.shape[1]))
    print(np.shape(XGlobal), np.shape(YGlobal), np.shape(XTestGlobal), np.shape(YTestGlobal))
    K.clear_session()

    cf_hyperparameters = {
        "unit": hp.choice("unit", [16, 32, 64]),
        "merge_mode": hp.choice("merge_mode", ['concat', 'sum', 'mul']),
        "batch": hp.choice("batch", [32, 64, 128]),
        'dropout': hp.uniform("dropout", 0, 0.8),
        "learning_rate": hp.uniform("learning_rate", 0.00001, 0.0001)
    }

    trials = Trials()
    # spark_trials = SparkTrials()
    fmin(hyperopt_cf, cf_hyperparameters, trials=trials, algo=tpe.suggest, max_evals=30)
    print('done')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Description of your script')
    # parser.add_argument('--dataset', type=str, default='UNSW_NB15', help='dataset name')
    #
    # args = parser.parse_args()
    train_DAE()
