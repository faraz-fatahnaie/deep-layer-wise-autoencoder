import os
import time

import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import load_model

from ACGAN import build_and_train_models
from Dataset2Image.Cart2Pixel import Cart2Pixel
from Dataset2Image.ConvPixel import ConvPixel
from autoencoder import Autoencoder, AutoencoderLayerWise
from utils import parse_data
from pathlib import Path


def deepinsight(param, config, save_path: Path):
    np.random.seed(param["seed"])
    time_file = open(save_path.joinpath('time.txt'), 'w')

    # FILES NAMING
    name = "_" + str(int(param["Max_A_Size"])) + "x" + str(int(param["Max_B_Size"]))
    if param["No_0_MI"]:
        name = name + "_No_0_MI"
    if param["mutual_info"]:
        name = name + "_MI"
    else:
        name = name + "_Mean"
    if param['autoencoder']:
        if param['ae_method'] == 'LW':
            name = name + '_AE_LW'
        elif param['ae_method'] == 'MAGNETO':
            name = name + '_AE_MAGNETO'

    filename_train = "train" + name + ".npy"
    filename_test = "test" + name + ".npy"

    if param['enhanced_dataset'] == 'gan':
        class_label = config['DEEPINSIGHT']['generate_class']
        ae_method = config['DEEPINSIGHT']['ae_method']
        image_size = str(int(config['DEEPINSIGHT']["Max_A_Size"])) + "x" + str(int(config['DEEPINSIGHT']["Max_B_Size"]))
        if class_label == 0:
            class_label_name = 'Normal'
        elif class_label == 1:
            class_label_name = 'Attack'
        if config['DEEPINSIGHT']['autoencoder']:
            model_name = f"acgan_aagm_ae_{ae_method}.h5"
            filename_X_gan = f'{class_label_name}_{image_size}_MI_AE_{ae_method}_gan.npy'
        else:
            model_name = "acgan_aagm.h5"
            filename_X_gan = f'{class_label_name}_{image_size}_MI_gan.npy'

    try:
        XGlobal = np.load(str(Path(config['DATASET_PATH']).joinpath(filename_train)))
        XTestGlobal = np.load(str(Path(config['DATASET_PATH']).joinpath(filename_test)))
        if param['enhanced_dataset'] == 'gan':
            X_gan = np.load(os.path.join(config['DATASET_PATH'], f'X_train_{filename_X_gan}'))
            y_gan = np.load(os.path.join(config['DATASET_PATH'], f'y_train_{filename_X_gan}'))
        print(f"TRAIN IMAGES LOADED WITH NAME OF {filename_train} AND SIZE OF {np.shape(XGlobal)}")
        print(f"TEST IMAGES LOADED WITH NAME OF {filename_test} AND SIZE OF {np.shape(XTestGlobal)}")
        return XGlobal, XTestGlobal

    except:
        train_df = pd.read_csv(Path(config['DATASET_PATH']).joinpath('train_' + config['CLASSIFICATION_MODE'] + '.csv'))
        x_train, y_train = parse_data(train_df, dataset_name=config['DATASET_NAME'], mode='df',
                                      classification_mode=config['CLASSIFICATION_MODE'])
        print(f'train shape: x=>{x_train.shape}, y=>{y_train.shape}')
        y_train = y_train.to_numpy()

        test_df = pd.read_csv(Path(config['DATASET_PATH']).joinpath('test_' + config['CLASSIFICATION_MODE'] + '.csv'))
        x_test, y_test = parse_data(test_df, dataset_name=config['DATASET_NAME'], mode='df',
                                    classification_mode=config['CLASSIFICATION_MODE'])
        y_test = y_test.to_numpy()
        print(f'test shape: x=>{x_test.shape}, y=>{y_test.shape}')

        if param['autoencoder']:
            ae_method = param['ae_method']
            ae_path = os.path.join(config['DATASET_PATH'], f'Autoencoder_{ae_method}.h5')
            try:
                ae = load_model(str(ae_path))
                print(f'AUTOENCODER LOADED FROM {ae_path}')
            except:
                if param['ae_method'] == 'MAGNETO':
                    ae_train_start_time = time.time()
                    ae = Autoencoder(x_train.to_numpy())
                    ae_train_end_time = time.time()
                    time_file.write(
                        f'Autoencoder {ae_method} Training (sec): {int(ae_train_end_time - ae_train_start_time)}\n')
                    time_file.write('\n')
                elif param['ae_method'] == 'LW':
                    ae_train_start_time = time.time()
                    ae = AutoencoderLayerWise(x_train.to_numpy())
                    ae_train_end_time = time.time()
                    time_file.write(
                        f'Autoencoder {ae_method} Training (sec): '
                        f'{int(ae_train_end_time - ae_train_start_time)}\n')
                ae.save(ae_path)
            x_train_cols = x_train.columns
            x_test_cols = x_test.columns
            ae_test_start_time = time.time()
            x_train = ae.predict(x_train)
            x_test = ae.predict(x_test)
            ae_test_end_time = time.time()
            time_file.write(f'Autoencoder {ae_method} Reconstructing (sec):'
                            f' {int(ae_test_end_time - ae_test_start_time)}\n')

            x_train = pd.DataFrame(x_train, columns=x_train_cols)
            x_test = pd.DataFrame(x_test, columns=x_test_cols)
            print(f'TRAIN AND TEST DATASETS RECONSTRUCTED USING {ae_method} AUTOENCODER.')

        try:
            XGlobal = np.load(str(Path(config['DATASET_PATH']).joinpath(filename_train)))
            XTestGlobal = np.load(str(Path(config['DATASET_PATH']).joinpath(filename_test)))
        except:
            image_encoding_start_time = time.time()
            print("transposing")
            # q["data"] is matrix T in paper (transpose of dataset without labels)
            # max_A_size, max_B_size is n and m in paper (the final size of generated image)
            # q["y"] is labels
            q = {"data": np.array(x_train.values).transpose(), "method": param["Method"],
                 "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": y_train.argmax(axis=-1)}
            print(q["method"])
            print(q["max_A_size"])
            print(q["max_B_size"])

            # generate images
            XGlobal, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], param["Dynamic_Size"],
                                                        mutual_info=param["mutual_info"], params=param,
                                                        only_model=False)

            # generate testing set image
            if param["mutual_info"]:
                x_test = x_test.drop(x_test.columns[toDelete], axis=1)

            x_test = np.array(x_test).transpose()
            print("generating Test Images for X_test with size ", x_test.shape)

            if image_model["custom_cut"] is not None:
                XTestGlobal = [ConvPixel(x_test[:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                         image_model["A"], image_model["B"],
                                         custom_cut=range(0, image_model["custom_cut"]))
                               for i in range(0, x_test.shape[1])]
            else:
                XTestGlobal = [ConvPixel(x_test[:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                         image_model["A"], image_model["B"])
                               for i in range(0, x_test.shape[1])]

            image_encoding_end_time = time.time()
            time_file.write(
                f'Image Encoding (sec): %d' % int(image_encoding_end_time - image_encoding_start_time))
            time_file.write('\n')

            np.save(str(Path(config['DATASET_PATH']).joinpath(filename_train)), XGlobal)
            print("Train Images generated and train images with labels are saved with the shape of:",
                  np.shape(XGlobal))
            np.save(str(Path(config['DATASET_PATH']).joinpath(filename_test)), XTestGlobal)
            print("Test Images generated and test images with labels are saved with the shape of:",
                  np.shape(XTestGlobal))

        if param['enhanced_dataset'] == 'gan':
            if not os.path.exists(os.path.join(config['DATASET_PATH'], f'X_train_{filename_X_gan}')):
                generator_path = os.path.join(config['DATASET_PATH'], model_name)
                try:
                    generator = load_model(generator_path)
                    print(f'GENERATOR LOADED FROM {generator_path}')
                except:
                    print('COULD NOT LOAD GENERATOR.')
                    gan_train_start_time = time.time()
                    build_and_train_models(config, np.array(XGlobal), y_train, np.array(XTestGlobal), y_test)
                    gan_train_end_time = time.time()
                    time_file.write(
                        f'GAN training (sec): %d' % int(gan_train_end_time - gan_train_start_time))
                    time_file.write('\n')

                    generator = load_model(generator_path)

                print(f'START GENERATING NEW SAMPLES OF {class_label_name} CLASS USING {model_name}')
                # generator = load_model(generator_path)

                generate_samples = config['DEEPINSIGHT']['generate_n_samples']

                noise_input = np.random.uniform(-1.0, 1.0, size=[generate_samples, 100])  # se 1 produce 1 sola immagine
                noise_label = np.zeros((generate_samples, 2))
                noise_label[:, class_label] = 1
                noise_input = [noise_input, noise_label]

                print('====== GENERATOR ARCHITECTURE ======')
                generator.summary()
                generator_start_time = time.time()
                predictions = generator.predict(noise_input)
                generator_end_time = time.time()
                time_file.write(
                    f'Generator {generator_start_time} Sampling (sec): %d' % int(generator_end_time - generator_start_time))
                time_file.write('\n')
                predictions = tf.reshape(predictions, [generate_samples, 10, 10])

                X_gan = predictions.numpy()
                y_gan = np.argmax(noise_label, axis=1)
                y_gan = np.reshape(y_gan, (-1, 1))

                np.save(str(Path(config['DATASET_PATH']).joinpath(f'X_train_{filename_X_gan}')), X_gan)
                np.save(str(Path(config['DATASET_PATH']).joinpath(f'y_train_{filename_X_gan}')), y_gan)
                print(f'SAVED GENERATED IMAGES WITH NAME OF X_train_{filename_X_gan} AND SHAPE OF {X_gan.shape}')
                print(f'SAVED GENERATED LABELS WITH NAME OF y_train_{filename_X_gan} AND SHAPE OF {y_gan.shape}')

        time_file.close()
        return np.array(XGlobal), np.array(XTestGlobal)

# if __name__ == '__main__':
#     deepinsight()
