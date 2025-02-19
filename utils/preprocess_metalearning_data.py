import numpy as np
from utils.noising_grids import create_fail
import random
import tensorflow as tf

def split_list(lst):
    # Remove list that are too small for meta-learning, and split the ones that are too big. We want a standard size
    if len(lst) <= 2:
        return [] 
    if len(lst) <= 6:
        return [lst]
    split_sizes = {6: [3, 3], 7: [4, 3], 8: [4, 4], 9: [5, 4], 10: [5, 5]}

    if len(lst) in split_sizes:
        sizes = split_sizes[len(lst)]
        parts = []
        index = 0
        for size in sizes:
            parts.append(lst[index:index + size])
            index += size
        return parts
    else:
        parts = []
        index = 0
        toggle = True 
        while len(lst) - index > 5:
            parts.append(lst[index:index + (4 if toggle else 5)])
            index += 4 if toggle else 5
            toggle = not toggle
        parts.append(lst[index:])
        return parts

def split_into_two_lists(processed_inputs):
    list_of_pairs = []
    remaining_list = []

    for sublist in processed_inputs:
        if len(sublist) >= 2:
            pair = sublist[:2]  # Extract only the first pair
            remainder = sublist[2:]  # Remaining elements after extracting one pair
            list_of_pairs.append(pair)
            if remainder:
                remaining_list.append(remainder)

    return list_of_pairs, remaining_list

def filter_and_split_inputs(inputs):

    filtered_and_split = []
    for sublist in inputs:
        if len(sublist) > 1:  # Remove lists with 1 or 2 elements
            filtered_and_split.extend(split_list(sublist))

    list_of_pairs, training_examples_d = split_into_two_lists(filtered_and_split)

    training_examples_labels = []
    training_examples_inputs = []
    for i, training_ex in enumerate(training_examples_d):
        training_ex, labels = create_fail(training_ex)
        training_examples_labels.append(labels)
        training_examples_inputs.append(training_ex)
    
    list_of_pairs = [sublist for sublist in list_of_pairs for _ in range(4)]

    training_examples_inputs = [item for sublist in training_examples_inputs for item in sublist]

    training_examples_labels = [item for sublist in training_examples_labels for item in sublist]

    shuffling = list(zip(list_of_pairs, training_examples_inputs, training_examples_labels))

    random.shuffle(shuffling)
    
    list_of_pairs, training_examples_inputs, training_examples_labels = zip(*shuffling)

    training_z, training_x, training_y =  np.array(list_of_pairs), np.array(training_examples_inputs), np.expand_dims(np.array(training_examples_labels), axis=-1)

    training_z_1, training_z_2 = tf.keras.ops.unstack(training_z, axis=1)


    training_z_1 = tf.squeeze(tf.concat(tf.keras.ops.unstack(training_z_1, axis=1), axis=-1))
    training_z_2 = tf.squeeze(tf.concat(tf.keras.ops.unstack(training_z_2, axis=1), axis=-1))

    training_x = tf.squeeze(tf.concat(tf.keras.ops.unstack(training_x, axis=1), axis=-1))


    return training_z_1, training_z_2, training_x, training_y
