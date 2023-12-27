from pathlib import Path

from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.python.client.session import InteractiveSession

from configs.setting import setting

from utils import set_seed

set_seed(0)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)

config, config_file = setting()

# Normal 0 / Intrusion 1 => class_label for Normal 0 and for Intrusion is 1
class_label = config['DEEPINSIGHT']['generate_class']

dataset_name = config['DATASET_NAME']
ae_method = config['DEEPINSIGHT']['ae_method']
base_path = config['DATASET_PATH']
image_size = str(int(config['DEEPINSIGHT']["Max_A_Size"])) + "x" + str(int(config['DEEPINSIGHT']["Max_B_Size"]))

if class_label == 0:
    class_label_name = 'Normal'
elif class_label == 1:
    class_label_name = 'Attack'

if config['DEEPINSIGHT']['autoencoder']:
    model_name = f"acgan_aagm_ae_{ae_method}.h5"
    out_name = f'{class_label_name}_{image_size}_MI_AE_{ae_method}_gan.npy'
else:
    model_name = "acgan_aagm.h5"
    out_name = f'{class_label_name}_{image_size}_MI_gan.npy'

generator = load_model(
    f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}\\{model_name}')

generate_samples = config['DEEPINSIGHT']['generate_n_samples']

noise_input = np.random.uniform(-1.0, 1.0, size=[generate_samples, 100])  # se 1 produce 1 sola immagine
noise_label = np.zeros((generate_samples, 2))
noise_label[:, class_label] = 1
noise_input = [noise_input, noise_label]

generator.summary()
predictions = generator.predict(noise_input)
predictions = tf.reshape(predictions, [generate_samples, 10, 10])

X_gan = predictions.numpy()
y_gan = np.argmax(noise_label, axis=1)
y_gan = np.reshape(y_gan, (-1, 1))

np.save(str(Path(base_path).joinpath(f'X_train_{out_name}')), X_gan)
np.save(str(Path(base_path).joinpath(f'y_train_{out_name}')), y_gan)
print(X_gan.shape)
print(y_gan.shape)
