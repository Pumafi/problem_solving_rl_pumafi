from tensorflow import keras
import pandas as pd
import numpy as np

import random
import math
from tqdm.notebook import trange, tqdm

import matplotlib.pyplot as plt
from matplotlib import colors


from scipy.stats import kde
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras import layers, losses
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

latent_dim = 256
class AutoEncoder(Model):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(30, 30, 10)),
      layers.Conv2D(64, (3, 3), activation='swish', padding='same', strides=2),
      #layers.Conv2D(128, (3, 3), activation='swish', padding='same', strides=1),
      #layers.Conv2D(128, (3, 3), activation='swish', padding='same', strides=1),
      #layers.Conv2D(256, (3, 3), activation='swish', padding='same', strides=1),
      layers.Dense(1024, activation=tf.keras.layers.LeakyReLU()),
      layers.Dropout(0.1),
      layers.Dense(1024, activation=tf.keras.layers.LeakyReLU()),
      layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
      layers.Dropout(0.1),
      layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
      layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
      layers.Dropout(0.1),
      layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
      layers.Dense(latent_dim, activation=tf.keras.layers.LeakyReLU()),])

    self.decoder = tf.keras.Sequential([
      layers.Dense(latent_dim, activation=tf.keras.layers.LeakyReLU()),
      layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
      layers.Dropout(0.1),
      layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
      layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
      layers.Dropout(0.1),
      layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
      layers.Dense(1024, activation=tf.keras.layers.LeakyReLU()),
      layers.Dropout(0.1),
      layers.Dense(1024, activation=tf.keras.layers.LeakyReLU()),
      #layers.Conv2D(256, (3, 3), activation='swish', padding='same', strides=1),
      #layers.Conv2DTranspose(128, kernel_size=3, strides=1, activation='swish', padding='same'),
      #layers.Conv2DTranspose(128, kernel_size=3, strides=1, activation='swish', padding='same'),
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='swish', padding='same'),
      layers.Conv2D(10, kernel_size=(3, 3), activation='softmax', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
