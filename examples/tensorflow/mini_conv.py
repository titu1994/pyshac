import os
import time
import numpy as np

import tensorflow as tf

import pyshac
from pyshac.core.managed.tf_engine import TensorflowSHAC

np.random.seed(0)
tf.set_random_seed(0)

"""
Finds good parameters for a basic_opt 2 layered MLP for fassion mnist.
"""


def prepare_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_parameters():
    """
    Setup the parameter search space.

    Returns:
        a list of hyper parameters
    """

    h1 = pyshac.DiscreteHP('h1', [8, 16, 32, 48])
    h2 = pyshac.DiscreteHP('h2', [8, 16, 32, 48])

    init1 = pyshac.DiscreteHP('init1', ['glorot_uniform', 'he_uniform'])
    init2 = pyshac.DiscreteHP('init2', ['glorot_uniform', 'he_uniform'])

    dropout1 = pyshac.UniformHP('dropout1', 0.1, 0.8)
    dropout2 = pyshac.UniformHP('dropout2', 0.1, 0.5)

    optimizer = pyshac.DiscreteHP('optimizer', ['adam', 'sgd'])
    learning_rate = pyshac.DiscreteHP('learning_rate', [1e-2, 5e-3, 1e-3])

    parameters = [h1, h2, init1, init2, dropout1, dropout2, optimizer, learning_rate]
    return parameters


def build_model(parameters):

    print("\n\nParameters : ", parameters, '\n\n')

    ip = tf.keras.layers.Input(shape=(784,))

    x = tf.keras.layers.Dense(parameters['h1'], activation='relu',
                              kernel_initializer=parameters['init1'])(ip)
    x = tf.keras.layers.Dropout(parameters['dropout1'])(x)

    x = tf.keras.layers.Dense(parameters['h2'], activation='relu',
                              kernel_initializer=parameters['init2'])(x)
    x = tf.keras.layers.Dropout(parameters['dropout2'])(x)

    x = tf.keras.layers.Dense(10, activation='softmax')(x)

    return ip, x


def evaluate_model(session, worker_id, parameters):
    """
    Evaluate a parameter setting

    # Arguments:
        session (tf.Session):
        worker_id (int):
        parameters (OrderedDict):

    # Returns:
        float value
    """
    ip, x = build_model(parameters)

    model = tf.keras.Model(ip, x)

    optimizer = parameters['optimizer']
    learning_rate = parameters['learning_rate']

    if optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    x_train, y_train, x_test, y_test = prepare_dataset()

    # train for just 2 epochs on mnist
    model.fit(x_train, y_train, batch_size=128, epochs=2, verbose=1, validation_data=(x_test, y_test))
    scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)

    # get the objective - test accuracy
    accuracy = float(scores[1])

    return accuracy


if __name__ == '__main__':
    # get the parameter configuration
    param_list = get_parameters()

    total_budget = 100  # train at maximum 50 models

    num_batches = 10  # train for 10 epochs : each of batchsize 10

    # number of gpus to use
    max_gpus = 1

    shac = TensorflowSHAC(evaluate_model, param_list, total_budget, num_batches=num_batches, objective='max',
                          max_gpu_evaluators=max_gpus, max_cpu_evaluators=num_batches)

    # If old training session exists, restore it.
    if os.path.exists('shac'):
        shac.restore_data()

    # too few samples per epoch to perform proper CV checks
    shac.fit(skip_cv_checks=True)

    # unnecessary since the dataset is saved during training
    # but good practice
    shac.save_data()

    # Use a different generator to predict 5 samples per batch instead of 10
    shac = TensorflowSHAC(evaluate_model, None, total_budget, num_batches=5, max_gpu_evaluators=0,
                          objective='max', max_cpu_evaluators=5)

    # load the data from training
    shac.restore_data()

    samples = shac.predict()

    # generated a batch of 5 classifiers which can now be evaluated
    for i, sample in enumerate(samples):
        print("Sample", i + 1, ": ", sample)