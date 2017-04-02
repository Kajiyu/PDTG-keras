# -*- coding: utf-8 -*-
# keras系
from keras import models
from keras import layers
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense,Merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.utils.visualize_util import plot
from keras.datasets import cifar100

# その他
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cPickle
import random
import sys
from tqdm import tqdm


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def create_generator(latent_size):
    latent_input = Input(shape=[latent_size])
    perception_input = Input(shape=[12])
    PH = Dense(800, init='glorot_normal')(perception_input)
    H = merge([latent_input, PH], mode='concat', concat_axis=1)
    H = BatchNormalization(mode=2)(H)
    H = Reshape( [5, 5, 40] )(H)
    H = Convolution2D(1024, 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = Activation('relu')(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(512, 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = Activation('relu')(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(256, 5, 5, border_mode='same', init='glorot_uniform')(H)
    H = Activation('relu')(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(128, 5, 5, border_mode='same', init='glorot_uniform')(H)
    H = Activation('relu')(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(64, 5, 5, border_mode='same', init='glorot_uniform')(H)
    H = Activation('relu')(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(32, 5, 5, border_mode='same', init='glorot_uniform')(H)
    H = Activation('relu')(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(1, 5, 5, border_mode='same', init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    generator_model = Model([latent_input, perception_input], g_V)
    return generator_model


def create_discriminator(img_shape):
    image_input = Input(shape=img_shape)
    perception_input = Input(shape=[12])
    H = Convolution2D(32, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(image_input)
    H = LeakyReLU(0.2)(H)
    H = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Convolution2D(1024, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Convolution2D(1024, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = Flatten()(H)
    H = Dense(100)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.5)(H)
    PH = Dense(100, init='glorot_normal')(perception_input)
    H = merge([H, PH], mode='sum')
    d_V = Dense(2,activation='softmax')(H)
    discriminator_model = Model([image_input, perception_input], d_V)
    return discriminator_model


# 論文にちゃんと書いてなかったぽん。
def create_perception_model(img_shape):
    image_input = Input(shape=img_shape)
    H = Convolution2D(32, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(image_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.5)(H)
    H = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.5)(H)
    H = Flatten()(H)
    H = Dense(500, init='glorot_normal')(H)
    H = Dense(500, init='glorot_normal')(H)
    H = Dense(12, init='glorot_normal')(H)
    p_V = Activation('sigmoid')(H)
    perception_model = Model(image_input, p_V)
    return perception_model


class PDTG:
    def __init__(self, latent_size, input_shape):
        self.latent_size = latent_size
        self.input_shape = input_shape
        self.is_model_created = False

    def create_model(self):
        self.generator = create_generator(self.latent_size)
        self.discriminator = create_discriminator(self.input_shape)
        self.perception_model = create_perception_model(self.input_shape)
        gan_input = Input(shape=[self.latent_size])
        perception_input = Input(shape=[12])
        H = self.generator([gan_input, perception_input])
        p_V = self.perception_model(H)
        g_V = self.discriminator([H, perception_input])
        self.g_p_model = Model([gan_input, perception_input], p_V)
        self.g_d_model = Model([gan_input, perception_input], g_V)
        self.model = Model([gan_input, perception_input], [g_V, p_V])
        self.g_p_model.summary()
        self.g_d_model.summary()
        plot(self.model, to_file="pdtg.png", show_shapes=True, show_layer_names=True)
        self.is_model_created = True

    def compile_model(self):
        self.generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))
        self.perception_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))
        self.g_d_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))
        self.g_p_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))
