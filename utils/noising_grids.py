import numpy as np
import tensorflow as tf
import random

g = tf.random.get_global_generator()


def cosine_schedule_function(t, max_t=1., epsilon=1e-3):
    t = tf.cast(t, tf.float32)
    f_t = tf.math.cos(((t / max_t) + epsilon) / (1 + epsilon) * (np.pi / 2)) ** 2
    f_0 = tf.math.cos(((t * 0 / max_t) + epsilon) / (1 + epsilon) * (np.pi / 2)) ** 2
    return f_t / f_0


def make_qmatrix_integral(nb_categories, t, tm1=None):
    Q_integral = tf.zeros((nb_categories, nb_categories), dtype=tf.float32)

    if tm1 is None:
        tm1 = tf.zeros_like(t, dtype=tf.float32)

    for i in range(nb_categories):
        for j in range(nb_categories):
            if i != j:
                Q_integral = tf.tensor_scatter_nd_update(Q_integral, indices=tf.constant([[i, j]]),
                                                         updates=[-1 / (nb_categories - 1) * tf.squeeze(
                                                             tf.math.log(cosine_schedule_function(t, max_t=1.)))
                                                                  + 1 / (nb_categories - 1) * tf.squeeze(
                                                             tf.math.log(cosine_schedule_function(tm1, max_t=1.)))])

    for i in range(nb_categories):
        Q_integral = tf.tensor_scatter_nd_update(Q_integral, indices=tf.constant([[i, i]]),
                                                 updates=[-tf.reduce_sum(Q_integral[i])])

    return Q_integral


def probability_function(nb_categories, t, step_size):
    """
    Compute the categorical forward probability from t to t + step_size

    """
    t = tf.cast(t, dtype=tf.float32)
    if step_size == 0.:
        tm1 = None
    else:
        tm1 = tf.cast(t - step_size, dtype=tf.float32)
    Q = make_qmatrix_integral(nb_categories, t, tm1)

    return Q


@tf.function
def forward_process(x0, step, nb_categories):
    if len(x0.shape) != 3:
        raise Exception("Entry must have 3 dims but {0} were found".format(len(x0.shape)))

    t = tf.squeeze(step)
    Q = tf.linalg.expm(make_qmatrix_integral(nb_categories, t))

    pxt = tf.tensordot(x0, Q, axes=[-1, 0])
    height, width, _ = pxt.shape
    flat_pxt = tf.reshape(pxt, (-1, nb_categories))
    cum_sum = tf.math.cumsum(flat_pxt, axis=-1)

    unif = tf.cast(g.uniform(shape=(len(cum_sum), 1)), dtype=tf.float32)
    random_values = tf.math.argmax((unif < cum_sum), axis=1)
    xt = tf.reshape(random_values, (height, width))
    xt = tf.cast(tf.one_hot(xt, nb_categories), dtype=tf.float32)

    return xt

def noise_grid(grid):
  k = random.randint(10, 15)
  t = 1 / k
  grid = forward_process(grid, t, 10).numpy()
  return grid

def color_change_grid(grid):
    k = random.randint(1, 5)
    grid = np.roll(grid, k, -1)
    return grid

def translate_grid(grid):
    axis = random.randint(1, 2)
    k = random.randint(3, 10)
    grid = np.roll(grid, k, axis)
    return grid

def flip_grid(grid):
    grid = np.flip(grid, axis=(0,1))
    return grid


def rotation(grid):
    k = random.randint(1, 3)
    grid = np.rot90(grid, k=k)
    return grid

def create_fail(challenge_grids):

    labels = [1] * len(challenge_grids)
    original_challenge_grid_length = len(challenge_grids)

    while len(challenge_grids) < 4:
        index_selected = random.randint(0, original_challenge_grid_length - 1)
        tuple_grids = challenge_grids[index_selected]
        grid = tuple_grids[1]

        failed_grid = grid

        while np.array_equal(failed_grid, grid):
            failure_case = random.randint(1, 5)
            match failure_case:
                case 1:
                    failed_grid = flip_grid (failed_grid)
                case 2:
                    failed_grid = rotation(failed_grid)
                case 3:
                    failed_grid = color_change_grid(failed_grid)
                case 4:
                    failed_grid = noise_grid(failed_grid)
                case 5:
                    failed_grid = translate_grid(failed_grid)

        challenge_grids.append((tuple_grids[0], failed_grid))
        labels.append(0)
    return challenge_grids, labels
