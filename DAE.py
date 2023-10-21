import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from model.ATTENTION import Attention1D
from model.CNNLSTM import CnnLstm
from model.DNN import DNN
from model.SE import SE
from sklearn import metrics
from keras.models import Model
from utils import parse_data
from model.CNN import CnnMagneto
from loss.EQLv2 import EQLv2
from loss.focal_loss import focal_loss
import os
from pathlib import Path
from configs.setting import setting
import json
from utils import OptimizerFactory, set_seed
from layer_wise_autoencoder import ae_factory, partial_ae_factory

print(keras.__version__)

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(device_lib.list_local_devices())

tf.compat.v1.disable_eager_execution()

set_seed()
i = 1
flag = True
SAVE_PATH_ = ''
TRAINED_MODEL_PATH_ = ''
CHECKPOINT_PATH_ = ''
config = {}
BASE_DIR = Path(__file__).resolve().parent
while flag:

    config, config_file = setting()
    TEMP_FILENAME = f"{config['DATASET_NAME']}-{config['CLASSIFICATION_MODE']}-{config['MODEL_NAME']}-{i}"
    TEMP_PATH = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

    if os.path.isdir(TEMP_PATH):
        i += 1
    else:
        flag = False

        os.mkdir(BASE_DIR.joinpath(f"session/{TEMP_FILENAME}"))
        SAVE_PATH_ = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

        os.mkdir(BASE_DIR.joinpath(f'{SAVE_PATH_}/model_checkpoint'))
        CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f"model_checkpoint/")

        with open(f'{SAVE_PATH_}/MODEL_CONFIG.json', 'w') as f:
            json.dump(config_file, f)

        print(f'MODEL SESSION: {SAVE_PATH_}')

# Generate synthetic data for testing (replace with your dataset)
dataset_name = config['DATASET_NAME']
train = pd.read_csv(
    f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}'
    f'\\train_binary_2neuron_labelOnehot.csv')
test = pd.read_csv(
    f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}'
    f'\\test_binary_2neuron_labelOnehot.csv')

X_train, y_train = parse_data(train, config['DATASET_NAME'], config['CLASSIFICATION_MODE'])
X_test, y_test = parse_data(test, config['DATASET_NAME'], config['CLASSIFICATION_MODE'])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

hidden_size = config['AE_HIDDEN_SIZE']
activation = config['AE_ACTIVATION']
loss_fn = config['AE_LOSS']
ae_epoch = config['AE_EPOCH']

ae_filename = f'AE_{dataset_name}_' + '_'.join(map(str, hidden_size)) + f'_{activation}_{loss_fn}_E{ae_epoch}'
ae_path = Path(f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\trained_ae\\{ae_filename}.keras')

if os.path.isfile(ae_path) and not config['AE_TRAINABLE']:
    deep_autoencoder = keras.models.load_model(ae_path)
    print(f'DAE loaded from {ae_path}')
else:
    # TODO: train partial ae independent from hidden_size
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

    # PRETRAIN STEP: AutoEncoder Layer-wise Training
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

    autoencoder1.fit(X_train, X_train,
                     epochs=config['AE_EPOCH'], batch_size=config['AE_BATCH_SIZE'],
                     # validation_split=0.20,
                     shuffle=False
                     )

    first_layer_code = encoder1.predict(X_train)
    autoencoder2.fit(first_layer_code, first_layer_code,
                     epochs=config['AE_EPOCH'], batch_size=config['AE_BATCH_SIZE'],
                     # validation_split=0.20,
                     shuffle=False
                     )

    second_layer_code = encoder2.predict(first_layer_code)
    autoencoder3.fit(second_layer_code, second_layer_code,
                     epochs=config['AE_EPOCH'], batch_size=config['AE_BATCH_SIZE'],
                     # validation_split=0.20,
                     shuffle=False
                     )

    # FINETUNE STEP
    # Setting the Weights of the Deep Autoencoder which has Learned in Layer-wise Training
    deep_autoencoder = ae_factory(in_shape=X_train.shape[1], hidden_size=hidden_size, activation=activation)
    # deep_autoencoder.compile(loss=config['AE_LOSS'], optimizer=sgd1)

    deep_autoencoder.layers[0].set_weights(autoencoder1.layers[2].get_weights())  # first dense layer
    deep_autoencoder.layers[1].set_weights(autoencoder1.layers[3].get_weights())  # first bn layer
    deep_autoencoder.layers[2].set_weights(autoencoder2.layers[2].get_weights())  # second dense layer
    deep_autoencoder.layers[3].set_weights(autoencoder2.layers[3].get_weights())  # second bn layer
    deep_autoencoder.layers[4].set_weights(autoencoder3.layers[2].get_weights())  # third dense layer
    deep_autoencoder.layers[5].set_weights(autoencoder3.layers[3].get_weights())  # third bn layer
    deep_autoencoder.layers[6].set_weights(autoencoder3.layers[4].get_weights())  # first decoder
    deep_autoencoder.layers[7].set_weights(autoencoder2.layers[4].get_weights())  # second decoder
    deep_autoencoder.layers[8].set_weights(autoencoder1.layers[4].get_weights())  # third decoder

    deep_autoencoder.save(SAVE_PATH_.joinpath("DAE.keras"))
    deep_autoencoder.save(ae_path)

# X_train_reconstructed = deep_autoencoder.predict(X_train)
# X_test_reconstructed = deep_autoencoder.predict(X_test)
# print(X_train_reconstructed.shape, y_train.shape)
# reconstructed_train_df = pd.DataFrame(np.concatenate((X_train_reconstructed, y_train), axis=1), columns=train.columns)
# reconstructed_test_df = pd.DataFrame(np.concatenate((X_test_reconstructed, y_test), axis=1), columns=test.columns)
# reconstructed_train_df.to_csv(os.path.join(SAVE_PATH_, 'train_ae' + '.csv'), index=False)
# reconstructed_test_df.to_csv(os.path.join(SAVE_PATH_, 'test_ae' + '.csv'), index=False)

classifier = {
    'DNN': DNN(in_shape=(X_train.shape[1],), out_shape=y_train.shape[1]).build_graph(),
    'ATTENTION': Attention1D(in_shape=(X_train.shape[1], 1), out_shape=y_train.shape[1]).build_graph(),
    'CNN': CnnMagneto(in_shape=(X_train.shape[1], 1), out_shape=y_train.shape[1]).build_graph(),
    'CNNLSTM': CnnLstm(in_shape=(X_train.shape[1], 1), out_shape=y_train.shape[1]).build_graph(),
    'SE': SE(in_shape=(X_train.shape[1], 1), out_shape=y_train.shape[1]).build_graph()

}

input_img = keras.Input(shape=(X_train.shape[1],))
ae_out = deep_autoencoder(input_img)
cf_out = classifier[config['MODEL_NAME']](ae_out)
model = Model(inputs=input_img, outputs=cf_out)

# opt_factory_cf = OptimizerFactory(opt=config['OPTIMIZER'],
#                                   lr_schedule=config['SCHEDULER'],
#                                   len_dataset=len(X_train),
#                                   epochs=config['EPOCHS'],
#                                   batch_size=config['BATCH_SIZE'],
#                                   init_lr=config['LR'],
#                                   final_lr=config['MIN_LR'])
#
# for layer in deep_autoencoder.layers[:]:
#     layer.trainable = False
#
# model.compile(loss='categorical_crossentropy', optimizer=opt_factory_cf.get_opt(), metrics=['accuracy'])
# model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath=CHECKPOINT_PATH_,
#     save_weights_only=True,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)
#
# model.fit(X_train, y_train,
#           validation_data=(X_test, y_test),
#           epochs=config['EPOCHS'],
#           batch_size=config['BATCH_SIZE'],
#           shuffle=True,
#           callbacks=[model_checkpoint])


for layer in deep_autoencoder.layers[:]:
    layer.trainable = True

opt_factory_cf = OptimizerFactory(opt=config['OPTIMIZER'],
                                  lr_schedule=config['SCHEDULER'],
                                  len_dataset=len(X_train),
                                  epochs=config['EPOCHS'],
                                  batch_size=config['BATCH_SIZE'],
                                  init_lr=config['LR'],
                                  final_lr=config['MIN_LR'])

classifier_loss_fn = {
    'cce': 'categorical_crossentropy',
    'bce': 'binary_crossentropy',
    'fcce': focal_loss(alpha=0.25),
    'eqloss': EQLv2()
}

model.compile(loss=classifier_loss_fn[config['LOSS']], optimizer=opt_factory_cf.get_opt(),
              metrics=['accuracy'])
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH_,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
csv_logger = tf.keras.callbacks.CSVLogger(f'{SAVE_PATH_}\\log.csv', append=True, separator=';')

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=config['EPOCHS'],
          batch_size=config['BATCH_SIZE'],
          shuffle=False,
          callbacks=[model_checkpoint, csv_logger])

deep_autoencoder.save(SAVE_PATH_.joinpath("DAE_FINETUNE.keras"))
model.load_weights(CHECKPOINT_PATH_)
# model = keras.models.load_model(CHECKPOINT_PATH_)

pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
y_eval = np.argmax(y_test, axis=1)
pred_df = pd.DataFrame({'y_pred': pred, 'y_true': y_eval})
pred_df.to_csv(SAVE_PATH_.joinpath(os.path.join('prediction' + '.csv')), index=False)

result = dict()
result['score'] = metrics.accuracy_score(y_eval, pred)
result['recall'] = metrics.recall_score(y_eval, pred, average='binary')
result['precision'] = metrics.precision_score(y_eval, pred, average='binary')
result['cm'] = metrics.confusion_matrix(y_eval, pred).tolist()
result['fpr'] = result['cm'][1][0] / (result['cm'][1][0] + result['cm'][1][1])
result['f1'] = metrics.f1_score(y_eval, pred, average='binary')

print(result)

with open(SAVE_PATH_.joinpath("result.txt"), 'w') as file:
    file.write(str(result))
