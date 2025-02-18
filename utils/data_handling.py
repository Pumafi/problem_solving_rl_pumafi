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

def get_color_map(number_of_categories=4):
    """
    Get the matplotlib colormap and norm for images visualisation
    Args:
        number_of_categories: number of facies in the slice

    Returns: cmap, norm

    """
    if number_of_categories == 4:
        cmap = colors.ListedColormap(["#FF8000", "#CBCB33", "#9898E5", "#66CB33"])
        bounds = [-0.1, 0.9, 1.9, 2.9, 3.9]
    elif number_of_categories == 5:
        cmap = colors.ListedColormap(["#000000", "#5387AD", "#7DD57E", "#F1E33E", "#C70000"])
        bounds = [-0.1, 0.9, 1.9, 2.9, 3.9, 4.9]
    else:  # 9
        cmap = colors.ListedColormap([
            "000000",   # Black
            "#1F77B4",  # Blue
            "#FF7F0E",  # Orange
            "#2CA02C",  # Green
            "#D62728",  # Red
            "#9467BD",  # Purple
            "#E377C2",  # Pink
            "#7F7F7F",  # Gray
            "#BCBD22",  # Yellow-Green
            "#17BECF"   # Cyan
        ])
        bounds = [-0.1, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9]

    norm = colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm

cmap, norm = get_color_map(number_of_categories=9)

def pad_to_shape(arr, target_shape=(30,30,1)):
    """
    Padding the inputs to a single shape, this will make it easier to manipulate
    """
    paddings = [(0, target_shape[i] - arr.shape[i]) for i in range(len(arr.shape))]

    padded_array = tf.pad(
        arr, paddings, mode='CONSTANT', constant_values=0
    )

    return padded_array

def preprocess_challenge_data(challenge_data, solution_data):
  challenge_ids = []

  # tuples (test_input, test_output) that are both inputs to solution propositioner
  challenge_propositioner_inputs = []

  # solver trainining input (might be useful)
  train_solver_inputs = []
  train_solver_outputs = []

  #solver test inputs (what the solver will train, getting also a solution as input)
  test_solver_inputs = []
  test_solver_outputs = []

  for id, challenge in challenge_data.items():
    challenge_ids.append(id)

    # TRAIN
    current_challenge_propositioner_inputs = []
    current_train_solver_inputs = []
    current_train_solver_outputs = []

    for train in challenge['train']:
      # input
      array = np.array(train['input'])

      if array.shape[-1] == 1:
        # Necessary or to_categorical will mess up the last dim
        array = np.expand_dims(array, axis=-1)
      array = pad_to_shape(array)
      input_cat_tensor = tf.keras.utils.to_categorical(array, num_classes=10)
      current_train_solver_inputs.append(input_cat_tensor)

      # output
      array = np.array(train['output'])

      if array.shape[-1] == 1:
        array = np.expand_dims(array, axis=-1)
      array = pad_to_shape(array)
      output_cat_tensor = tf.keras.utils.to_categorical(array, num_classes=10)
      current_train_solver_outputs.append(output_cat_tensor)

      current_challenge_propositioner_inputs.append((input_cat_tensor, output_cat_tensor))

    challenge_propositioner_inputs.append(current_challenge_propositioner_inputs)
    train_solver_inputs.append(current_train_solver_inputs)
    train_solver_outputs.append(current_train_solver_outputs)

    # test
    current_test_solver_inputs = []
    current_test_solver_outputs = []
    for i, test in enumerate(challenge['test']):
      # TEST INPUTS
      array = np.array(test['input'])

      if array.shape[-1] == 1:
        array = np.expand_dims(array, axis=-1)
      array = pad_to_shape(array)
      input_cat_tensor = tf.keras.utils.to_categorical(array, num_classes=10)
      current_test_solver_inputs.append(input_cat_tensor)

      # TEST OUTPUTS
      array = np.array(solution_data[id][i])

      if array.shape[-1] == 1:
        array = np.expand_dims(array, axis=-1)
      array = pad_to_shape(array)
      output_cat_tensor = tf.keras.utils.to_categorical(array, num_classes=10)

      current_test_solver_outputs.append(output_cat_tensor)

    current_test_solver_inputs = np.array(current_test_solver_inputs)
    test_solver_inputs.append(current_test_solver_inputs)

    current_test_solver_outputs = np.array(current_test_solver_outputs)
    test_solver_outputs.append(current_test_solver_outputs)

  return challenge_propositioner_inputs, train_solver_inputs, train_solver_outputs, test_solver_inputs, test_solver_outputs
