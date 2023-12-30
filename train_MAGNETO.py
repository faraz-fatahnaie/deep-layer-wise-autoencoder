import csv
import time

import keras.layers
import pandas as pd
import numpy as np
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import json
from pathlib import Path

from keras import backend as K
from utils import parse_data, result
from configs.setting import setting
from Dataset2Image.main import deepinsight
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten

from utils import set_seed, get_result

# Set GPU device and disable eager execution
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.compat.v1.disable_eager_execution()
print(tf.config.list_physical_devices('GPU'))

set_seed(seed=0)

XGlobal = []
YGlobal = []

XTestGlobal = []
YTestGlobal = []

SavedParameters = []
Mode = ""
Name = ""
SAVE_PATH_ = ""
CHECKPOINT_PATH_ = ""
best_val_acc = 0

attack_label = 0
tid = 0


def CNN2(images, y, params=None):
    global tid

    print(params)
    K.clear_session()

    tid += 1
    x_train, x_test, y_train, y_test = train_test_split(images,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=0
                                                        )
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    image_size = x_train.shape[1]
    image_size2 = x_train.shape[2]

    x_train = np.reshape(x_train, [-1, image_size, image_size2, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size2, 1])

    kernel = params["kernel"]
    inputs = Input(shape=(image_size, image_size2, 1))

    X = Conv2D(32, (kernel, kernel),
               activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
               bias_initializer=tf.keras.initializers.Zeros(),
               name='conv0')(inputs)
    X = Dropout(rate=params['dropout1'])(X)
    X = Conv2D(64, (kernel, kernel),
               activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
               bias_initializer=tf.keras.initializers.Zeros(),
               name='conv1')(X)
    X = Dropout(rate=params['dropout2'])(X)
    X = Conv2D(128, (kernel, kernel),
               activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
               bias_initializer=tf.keras.initializers.Zeros(),
               name='conv2')(X)
    X = Flatten()(X)
    X = Dense(256,
              activation='relu',
              kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
              bias_initializer=tf.keras.initializers.Zeros())(X)
    X = Dense(1024,
              activation='relu',
              kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
              bias_initializer=tf.keras.initializers.Zeros())(X)
    X = Dense(2,
              activation='softmax',
              kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
              bias_initializer=tf.keras.initializers.Zeros())(X)

    model = Model(inputs, X)
    adam = Adam(params["learning_rate"])

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    # Train the model.
    train_start_time = time.time()
    model.fit(
        x_train,
        y_train,
        epochs=params["epoch"],
        verbose=2,
        validation_data=(x_test, y_test),
        batch_size=params["batch"],
        callbacks=[
            EarlyStopping(monitor='val_loss',
                          mode='min',
                          patience=10),
            ModelCheckpoint(filepath=os.path.join(CHECKPOINT_PATH_, 'best_model.h5'),
                            monitor='val_loss',
                            save_best_only=True)
        ]
    )
    train_end_time = time.time()

    model.load_weights(os.path.join(CHECKPOINT_PATH_, 'best_model.h5'))
    Y_predicted = model.predict(x_test, verbose=0, use_multiprocessing=True, workers=12)

    y_test = np.argmax(y_test, axis=1)
    Y_predicted = np.argmax(Y_predicted, axis=1)

    cm_val = confusion_matrix(y_test, Y_predicted)
    results_val = get_result(cm_val)

    return model, {"tid": tid,
                   "train_time": int(train_end_time - train_start_time),
                   "kernel": params["kernel"],
                   "learning_rate": params["learning_rate"],
                   "batch": params["batch"],
                   "dropout1": params["dropout1"],
                   "dropout2": params["dropout2"],
                   "TP_val": cm_val[0][0],
                   "FP_val": cm_val[0][1],
                   "TN_val": cm_val[1][1],
                   "FN_val": cm_val[1][0],
                   "OA_val": results_val['OA'],
                   "P_val": results_val['P'],
                   "R_val": results_val['R'],
                   "F1_val": results_val['F1'],
                   "FAR_val": results_val['FAR'],
                   }


def hyperopt_fcn(params):
    global SavedParameters
    global best_val_acc

    print("start train")
    train_start_time = time.time()
    model, val = CNN2(XGlobal, YGlobal, params)
    train_elapsed_time = time.time() - train_start_time
    print("start predict")
    test_start_time = time.time()
    y_predicted = model.predict(XTestGlobal, verbose=0, workers=4)
    test_elapsed_time = time.time() - test_start_time

    y_predicted = np.argmax(y_predicted, axis=1)
    YTestGlobal_temp = np.argmax(YTestGlobal, axis=1)

    cm_test = confusion_matrix(YTestGlobal_temp, y_predicted)
    results_test = get_result(cm_test)

    SavedParameters.append(val)
    SavedParameters[-1].update({
        "test_time": int(test_elapsed_time),
        "TP_test": cm_test[0][0],
        "FP_test": cm_test[0][1],
        "FN_test": cm_test[1][0],
        "TN_test": cm_test[1][1],
        "OA_test": results_test['OA'],
        "P_test": results_test['P'],
        "R_test": results_test['R'],
        "F1_test": results_test['F1'],
        "FAR_test": results_test['FAR'],
    })

    # Save model
    if SavedParameters[-1]["F1_val"] > best_val_acc:
        print("new saved model:" + str(SavedParameters[-1]))
        model.save(os.path.join(SAVE_PATH_, Name.replace(".csv", "_model.h5")))
        best_val_acc = SavedParameters[-1]["F1_val"]

    SavedParameters = sorted(SavedParameters, key=lambda i: i['F1_val'], reverse=True)

    try:
        with open((os.path.join(SAVE_PATH_, 'best_result.csv')), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")
    return {'loss': -val["F1_val"], 'status': STATUS_OK}


def train_MAGNETO():
    global YGlobal
    global YTestGlobal
    global XGlobal
    global XTestGlobal

    i = 1
    flag = True
    project = 'MAGNETO'
    global SAVE_PATH_
    TRAINED_MODEL_PATH_ = ''
    global CHECKPOINT_PATH_
    config = {}
    BASE_DIR = Path(__file__).resolve().parent
    while flag:

        config, config_file = setting(project=project)
        TEMP_FILENAME = f"{config['DATASET_NAME']}-{config['CLASSIFICATION_MODE']}-CNN-MAGNETO-{i}"
        TEMP_PATH = BASE_DIR.joinpath(f"session_{project}/{TEMP_FILENAME}")

        if os.path.isdir(TEMP_PATH):
            i += 1
        else:
            flag = False

            os.mkdir(BASE_DIR.joinpath(f"session_{project}/{TEMP_FILENAME}"))
            SAVE_PATH_ = BASE_DIR.joinpath(f"session_{project}/{TEMP_FILENAME}")

            os.mkdir(BASE_DIR.joinpath(f'{SAVE_PATH_}/model_checkpoint'))
            CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f"model_checkpoint")

            with open(f'{SAVE_PATH_}/CONFIG.json', 'w') as f:
                json.dump(config_file, f)

            print(f'MODEL SESSION: {SAVE_PATH_}')

    # Load and preprocess the training and testing data
    dataset_name = config['DATASET_NAME']
    train = pd.read_csv(
        f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}\\train_binary.csv')
    test = pd.read_csv(
        f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}\\test_binary.csv')

    if config['DEEPINSIGHT']['deepinsight']:
        XGlobal, XTestGlobal = deepinsight(config['DEEPINSIGHT'], config, SAVE_PATH_)
        _, YGlobal = parse_data(train, dataset_name=config['DATASET_NAME'], mode='np',
                                classification_mode=config['CLASSIFICATION_MODE'])
        _, YTestGlobal = parse_data(test, dataset_name=config['DATASET_NAME'], mode='np',
                                    classification_mode=config['CLASSIFICATION_MODE'])
        if config['DEEPINSIGHT']['enhanced_dataset'] == 'gan':
            class_label = config['DEEPINSIGHT']['generate_class']
            ae_method = config['DEEPINSIGHT']['ae_method']
            image_size = str(int(config['DEEPINSIGHT']["Max_A_Size"])) + "x" + str(
                int(config['DEEPINSIGHT']["Max_B_Size"]))
            if class_label == 0:
                class_label_name = 'Normal'
            elif class_label == 1:
                class_label_name = 'Attack'
            if config['DEEPINSIGHT']['autoencoder']:
                out_name = f'{class_label_name}_{image_size}_MI_AE_{ae_method}_gan.npy'
            else:
                out_name = f'{class_label_name}_{image_size}_MI_gan.npy'
            X_gan = np.load(os.path.join(config['DATASET_PATH'], f'X_train_{out_name}'))
            y_gan = np.load(os.path.join(config['DATASET_PATH'], f'y_train_{out_name}'))
            XGlobal = np.concatenate((XGlobal, X_gan), axis=0)
            YGlobal = np.concatenate((YGlobal, y_gan), axis=0)
            print('ORIGINAL DATASET MERGE WITH GAN GENERATED SAMPLES')

    else:
        XGlobal, YGlobal = parse_data(train, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                                      classification_mode=config['CLASSIFICATION_MODE'])
        XTestGlobal, YTestGlobal = parse_data(test, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                                              classification_mode=config['CLASSIFICATION_MODE'])

    # Reshape and preprocess the data
    YGlobal = keras.utils.to_categorical(YGlobal, num_classes=2)
    YTestGlobal = keras.utils.to_categorical(YTestGlobal, num_classes=2)
    XGlobal = np.reshape(XGlobal, (-1, 10, 10, 1))
    XTestGlobal = np.reshape(XTestGlobal, (-1, 10, 10, 1))
    print(np.shape(XGlobal), np.shape(YGlobal), np.shape(XTestGlobal), np.shape(YTestGlobal))

    hyperparameters_to_optimize = {"kernel": hp.choice("kernel", np.arange(2, 4 + 1)),
                                   "batch": hp.choice("batch", [32, 64, 128, 256, 512]),
                                   'dropout1': hp.uniform("dropout1", 0, 1),
                                   'dropout2': hp.uniform("dropout2", 0, 1),
                                   "learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),
                                   "epoch": 150}

    trials = Trials()
    fmin(hyperopt_fcn, hyperparameters_to_optimize,
         trials=trials,
         algo=tpe.suggest,
         max_evals=30,
         rstate=np.random.default_rng(0))
    print('done')


if __name__ == '__main__':
    train_MAGNETO()
