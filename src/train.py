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

from .model import PDTG


def plot_loss(losses):
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g_d"], label='generative-discriminitive loss')
        plt.plot(losses["g_p"], label='generative-perspective loss')
        plt.plot(losses["p"], label='perspective loss')
        plt.legend()
        plt.show()


def plot_gen(n_ex=16,dim=(4,4), figsize=(10,10) ):
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def train(pdtg, batch_size, nb_epoch):
    losses = {"d":[], "g_d":[], "g_p":[], "p":[]}
    index = 1
    for e in tqdm(range(nb_epoch)):
        print "--------------%d回目のTraining--------------" % index
        #batchの作成
        indecies = np.random.randint(0,X_train.shape[0],size=batch_size)
        image_batch = X_train[indecies,:,:,:]
        perception_features = df.iloc[indecies].values
        noise_gen = np.random.uniform(0,1,size=[batch_size,200])
        noise_gen_per = np.random.uniform(0,1,size=[batch_size,12])
        generated_images = pdtg.generator.predict([noise_gen, noise_gen_per])

        #ジェネった画像でdiscriminatorの学習
        X = np.concatenate((image_batch, generated_images))
        X_2 = np.concatenate((perception_features, noise_gen_per))
        y = np.zeros([2*batch_size,2])
        y[0:batch_size,1] = 1
        y[batch_size:,0] = 1
        d_loss  = pdtg.discriminator.train_on_batch([X, X_2],y)
        losses["d"].append(d_loss)
        print "Training Discriminator :::: value : %f" % d_loss

        #ジェネった画像でperception_modelの学習
        p_loss = pdtg.perception_model.train_on_batch(X, X_2)
        losses["p"].append(p_loss)
        print "Training Perception Model :::: value : %f" % p_loss

        noise_tr = np.random.uniform(0,1,size=[batch_size,200])
        noise_tr_per = np.random.uniform(0,1,size=[batch_size,12])
        y2 = np.zeros([batch_size,2])
        y2[:,1] = 1

        g_d_loss = pdtg.g_d_model.train_on_batch([noise_tr, noise_tr_per],y2)
        losses["g_d"].append(g_d_loss)
        print "Training Generative Model(Dis) :::: value : %f" % g_d_loss
        g_p_loss = pdtg.g_p_model.train_on_batch([noise_tr, noise_tr_per],noise_gen_per)
        losses["g_p"].append(g_p_loss)
        print "Training Generative Model(Per) :::: value : %f" % g_p_loss
        print "\n\n\n"
        index = index + 1
        # Updates plots
        if e%25==24:
            plot_loss(losses)


params = [
    "contrast",
    "repetitive",
    "granular",
    "random",
    "rough",
    "feature density",
    "direction",
    "structural complexity",
    "coarse",
    "regular",
    "oriented",
    "uniform"
]
df = pd.read_csv('texture_database/rating.csv')

max_val = 0.0
min_val = 5.0
for param in params:
    for val in df[param]:
        if max_val < val:
            max_val = val
        if min_val > val:
            min_val = val
print "max : %f, min %f" % (max_val, min_val)
range_num = max_val - min_val
for param in params:
    index = 0
    for val in df[param]:
        df[param][index] = (val - min_val) / range_num
        index = index + 1
#     print df[param]

# 画像を変換
images = []
for i in range(450):
    index = i + 1
    img_path = "./texture_database/textures/" + str(index) + ".png"
    img = load_img(img_path, target_size=(320,320), grayscale=True)
    arr = np.asarray(img)
    arr = arr.tolist()
    image = []
    image.append(arr)
    images.append(image)
images = np.array(images)
print images.shape


images = images.reshape([450, 320, 320,1])
X_train, X_test = np.vsplit(images, [400])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print np.min(X_train), np.max(X_train)
print('X_train shape:', X_train.shape)
print('Image Shape:', X_train.shape[1:])
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')



pdtg = PDTG(200, (320, 320, 1))
pdtg.create_model()
pdtg.compile_model()
train(pdtg, 100, 5000)
