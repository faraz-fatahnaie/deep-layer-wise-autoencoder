'''GAN model builder and util functions

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

'''
import keras
import pandas as pd
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.models import Model

import numpy as np
import math
import matplotlib.pyplot as plt
import os

from keras.optimizers import RMSprop
from keras.utils import to_categorical

from configs.setting import setting
from utils import parse_data
# from Dataset2Image.main import deepinsight

from utils import set_seed

set_seed(0)


def generator(inputs,
              image_size,
              activation='sigmoid',
              labels=None,
              codes=None):
    """Build a Generator Model

    Stack of BN-ReLU-Conv2DTranpose to generate fake images.
    Output activation is sigmoid instead of tanh in [1].
    Sigmoid converges easily.

    Arguments:
        inputs (Layer): Input layer of the generator (the z-vector)
        image_size (int): Target size of one side
            (assuming square image)
        activation (string): Name of output activation layer
        labels (tensor): Input labels
        codes (list): 2-dim disentangled codes for InfoGAN

    Returns:
        Model: Generator Model
    """
    image_resize = image_size // 2
    # network parameters
    kernel_size = 5
    layer_filters = [256, 128, 64, 32, 1]

    if labels is not None:
        if codes is None:
            # ACGAN labels
            # concatenate z noise vector and one-hot labels
            inputs = [inputs, labels]
        else:
            # infoGAN codes
            # concatenate z noise vector,
            # one-hot labels and codes 1 & 2
            inputs = [inputs, labels] + codes
        x = concatenate(inputs, axis=1)
    elif codes is not None:
        # generator 0 of StackedGAN
        inputs = [inputs, codes]
        x = concatenate(inputs, axis=1)
    else:
        # default input is just 100-dim noise (z-code)
        x = inputs

    x = Dense(image_resize * image_resize * layer_filters[0])(x)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        # first two convolution layers use strides = 2
        # the last two use strides = 1
        if filters > layer_filters[-4]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)

    if activation is not None:
        x = Activation(activation)(x)

    # generator output is the synthesized image x
    return Model(inputs, x, name='generator')


def discriminator(inputs,
                  activation='sigmoid',
                  num_labels=None,
                  num_codes=None):
    """Build a Discriminator Model

    Stack of LeakyReLU-Conv2D to discriminate real from fake
    The network does not converge with BN so it is not used here
    unlike in [1]

    Arguments:
        inputs (Layer): Input layer of the discriminator (the image)
        activation (string): Name of output activation layer
        num_labels (int): Dimension of one-hot labels for ACGAN & InfoGAN
        num_codes (int): num_codes-dim Q network as output
                    if StackedGAN or 2 Q networks if InfoGAN


    Returns:
        Model: Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
    for filters in layer_filters:
        # first 3 convolution layers use strides = 2
        # last one uses strides = 1
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    # default output is probability that the image is real
    outputs = Dense(1)(x)
    if activation is not None:
        print(activation)
        outputs = Activation(activation)(outputs)

    if num_labels:
        # ACGAN and InfoGAN have 2nd output
        # 2nd output is 10-dim one-hot vector of label
        layer = Dense(layer_filters[-2])(x)
        labels = Dense(num_labels)(layer)
        labels = Activation('softmax', name='label')(labels)
        if num_codes is None:
            outputs = [outputs, labels]
        else:
            # InfoGAN have 3rd and 4th outputs
            # 3rd output is 1-dim continous Q of 1st c given x
            code1 = Dense(1)(layer)
            code1 = Activation('sigmoid', name='code1')(code1)

            # 4th output is 1-dim continuous Q of 2nd c given x
            code2 = Dense(1)(layer)
            code2 = Activation('sigmoid', name='code2')(code2)

            outputs = [outputs, labels, code1, code2]
    elif num_codes is not None:
        # StackedGAN Q0 output
        # z0_recon is reconstruction of z0 normal distribution
        z0_recon = Dense(num_codes)(x)
        z0_recon = Activation('tanh', name='z0')(z0_recon)
        outputs = [outputs, z0_recon]

    return Model(inputs, outputs, name='discriminator')


def train(models, data, params, gan_save_path):
    """Train the discriminator and adversarial Networks

    Alternately train discriminator and adversarial
    networks by batch.
    Discriminator is trained first with real and fake
    images and corresponding one-hot labels.
    Adversarial is trained next with fake images pretending
    to be real and corresponding one-hot labels.
    Generate sample images per save_interval.

    # Arguments
        models (list): Generator, Discriminator,
            Adversarial models
        data (list): x_train, y_train data
        params (list): Network parameters

    """
    # the GAN models
    generator_gan, discriminator_gan, adversarial = models
    # images and their one-hot labels
    x_train, y_train = data
    # network parameters
    batch_size, latent_size, train_steps, num_labels, model_name = params
    # the generator image is saved every 500 steps
    save_interval = 500
    # noise vector to see how the generator
    # output evolves during training
    noise_input = np.random.uniform(-1.0,
                                    1.0,
                                    size=[16, latent_size])
    # class labels are 0, 1,
    # the generator must produce these classes
    noise_label = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    # number of elements in train dataset
    train_size = x_train.shape[0]
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_label, axis=1))

    for i in range(train_steps):
        # train the discriminator for 1 batch
        # 1 batch of real (label=1.0) and fake images (label=0.0)
        # randomly pick real images and
        # corresponding labels from dataset
        rand_indexes = np.random.randint(0,
                                         train_size,
                                         size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]
        # generate fake images from noise using generator
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        # randomly pick one-hot labels
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        # generate fake images
        fake_images = generator_gan.predict([noise, fake_labels])
        # real + fake images = 1 batch of train data
        x = np.concatenate((real_images, fake_images))
        # real + fake labels = 1 batch of train data labels
        labels = np.concatenate((real_labels, fake_labels))

        # label real and fake images
        # real images label is 1.0
        y = np.ones([2 * batch_size, 1])
        # fake images label is 0.0
        y[batch_size:, :] = 0
        # train discriminator network, log the loss and accuracy
        # ['loss', 'activation_1_loss',
        # 'label_loss', 'activation_1_acc', 'label_acc']
        metrics = discriminator_gan.train_on_batch(x, [y, labels])
        fmt = "%d: [disc loss: %f, srcloss: %f,"
        fmt += "lblloss: %f, srcacc: %f, lblacc: %f]"
        log = fmt % (i, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])

        # train the adversarial network for 1 batch
        # 1 batch of fake images with label=1.0 and
        # corresponding one-hot label or class
        # since the discriminator weights are frozen
        # in adversarial network only the generator is trained
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        # randomly pick one-hot labels
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        # label fake images as real
        y = np.ones([batch_size, 1])
        # train the adversarial network
        # note that unlike in discriminator training,
        # we do not save the fake images in a variable
        # the fake images go to the discriminator input
        # of the adversarial for classification
        # log the loss and accuracy
        metrics = adversarial.train_on_batch([noise, fake_labels],
                                             [y, fake_labels])
        fmt = "%s [advr loss: %f, srcloss: %f,"
        fmt += "lblloss: %f, srcacc: %f, lblacc: %f]"
        log = fmt % (log, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])
        print(log)
        # if (i + 1) % save_interval == 0:
        #     # plot generator images on a periodic basis
        #     plot_images(generator_gan,
        #                 noise_input=noise_input,
        #                 noise_label=noise_label,
        #                 show=False,
        #                 step=(i + 1),
        #                 model_name=model_name)

    # save the model after training the generator
    # the trained generator can be reloaded
    # for future MNIST digit generation

    generator_gan.save(str(os.path.join(gan_save_path, f'{model_name}.h5')))


def build_and_train_models(config=None, X_train=None, y_train=None, X_test=None, y_test=None):
    """Load the dataset, build ACGAN discriminator,
    generator, and adversarial models.
    Call the ACGAN train routine.
    """
    print('START GAN TRAINING ...')
    # config, config_file = setting()
    # dataset_name = config['DATASET_NAME']
    # if X_train is None and y_train is None and X_test is None and y_test is None:
    #     train_df = pd.read_csv(
    #         f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}'
    #         f'\\train_binary.csv')
    #     test_df = pd.read_csv(
    #         f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\{dataset_name}'
    #         f'\\test_binary.csv')
    #
    #     X_train, X_test = deepinsight(config['DEEPINSIGHT'], config)
    #     _, y_train = parse_data(train_df, dataset_name=dataset_name, classification_mode='binary')
    #     _, y_test = parse_data(test_df, dataset_name=dataset_name, classification_mode='binary')

    num_labels = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    image_size = X_train.shape[1]
    # print('image size', image_size)
    x_train = np.reshape(X_train,
                         [-1, image_size, image_size, 1])

    if config['DEEPINSIGHT']['autoencoder']:
        ae_method = config['DEEPINSIGHT']['ae_method']
        model_name = f"acgan_aagm_ae_{ae_method}"
    else:
        model_name = "acgan_aagm"
    # network parameters
    latent_size = 100
    batch_size = 64
    train_steps = 1000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels,)

    # build discriminator Model
    inputs = Input(shape=input_shape,
                   name='discriminator_input')
    # call discriminator builder
    # with 2 outputs, pred source and labels
    dis_gan = discriminator(inputs,
                            num_labels=num_labels)
    # [1] uses Adam, but discriminator
    # easily converges with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    # 2 loss fuctions: 1) probability image is real
    # 2) class label of the image
    loss = ['binary_crossentropy', 'categorical_crossentropy']
    dis_gan.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=['accuracy'])
    dis_gan.summary()

    # build generator model
    input_shape = (latent_size,)
    inputs = Input(shape=input_shape, name='z_input')
    labels = Input(shape=label_shape, name='labels')
    # call generator builder with input labels
    gen_gan = generator(inputs,
                        image_size,
                        labels=labels)
    gen_gan.summary()

    # build adversarial model = generator + discriminator
    optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    # freeze the weights of discriminator
    # during adversarial training
    dis_gan.trainable = False
    adversarial = Model([inputs, labels],
                        dis_gan(gen_gan([inputs, labels])),
                        name=model_name)
    # same 2 loss fuctions: 1) probability image is real
    # 2) class label of the image
    adversarial.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    # adversarial.summary()

    # train discriminator and adversarial networks
    models = (gen_gan, dis_gan, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size,
              train_steps, num_labels, model_name)

    # print(num_labels)

    # print(x_train.shape)
    # print(image_size)
    train(models, data, params, config['DATASET_PATH'])
    print('END OF GAN TRAINING ...')


def plot_images(generator_gan,
                noise_input,
                noise_label=None,
                noise_codes=None,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake images and plot them

    For visualization purposes, generate fake images
    then plot them in a square grid

    # Arguments
        generator (Model): The Generator Model for
            fake images generation
        noise_input (ndarray): Array of z-vectors
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images
        model_name (string): Model name

    """
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    print(filename)
    rows = int(math.sqrt(noise_input.shape[0]))
    if noise_label is not None:
        noise_input = [noise_input, noise_label]
        if noise_codes is not None:
            noise_input += noise_codes

    images = generator_gan.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


def test_generator(generator_gan):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    plot_images(generator_gan,
                noise_input=noise_input,
                show=True,
                model_name="test_outputs")


if __name__ == '__main__':
    build_and_train_models()
