import os
import shutil
import six
import time
import pytest
import warnings
import numpy as np

from pyshac.config import hyperparameters as hp, data
from pyshac.core.managed import keras_engine as engine

import tensorflow as tf

warnings.simplefilter('ignore', DeprecationWarning)


# wrapper function to clean up saved files and be deterministic
def tf_optimizer_wrapper(func):
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        np.random.seed(0)
        tf.set_random_seed(0)

        output = func(*args, **kwargs)

        np.random.seed(None)
        tf.set_random_seed(None)

        # remove temporary files
        if os.path.exists('shac/'):
            shutil.rmtree('shac/')

        if os.path.exists('custom/'):
            shutil.rmtree('custom/')

        return output
    return wrapper


def get_hyperparameter_list():
    h1 = hp.DiscreteHyperParameter('h1', [0, 1, 2])
    h2 = hp.DiscreteHyperParameter('h2', [3, 4, 5, 6])
    h3 = hp.UniformContinuousHyperParameter('h3', 1, 25)
    h4 = hp.DiscreteHyperParameter('h4', ['v1', 'v2'])
    return [h1, h2, h3, h4]


def get_hartmann6_hyperparameter_list():
    h = [hp.UniformContinuousHyperParameter('h%d' % i, 0.0, 1.0) for i in range(6)]
    return h


def get_branin_hyperparameter_list():
    h1 = hp.UniformContinuousHyperParameter('h1', -5.0, 10.0)
    h2 = hp.UniformContinuousHyperParameter('h2', 0.0, 15.0)
    return [h1, h2]


""" Numpy Evaluators """


def evaluation_simple(worker_id, params):
    values = list(params.values())[:3]
    metric = np.sum(values)
    print('objective value =', metric)
    return metric


def evaluation_hartmann6(worker_id, params):
    """ Code ported from https://www.sfu.ca/~ssurjano/Code/hart6scm.html
    Global Minimum = -3.32237
    """

    xx = np.array(list(params.values()), dtype=np.float32)

    alpha = np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float32)

    A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], dtype=np.float32)

    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]], dtype=np.float64)

    xx = xx.reshape((6, 1))

    inner = np.sum(A * ((xx.T - P) ** 2), axis=-1)
    inner = np.exp(-inner)
    outer = np.sum(alpha * inner)

    return -outer


def evaluation_branin(worker_id, params):
    """ Code ported from https://www.sfu.ca/~ssurjano/Code/braninm.html
    Global Minimum = -0.397887
    """

    xx = list(params.values())
    x1, x2 = xx[0], xx[1]

    a = 1.0
    b = 5.1 / (4 * (np.pi ** 2))
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    term1 = a * ((x2 - b * (x1 ** 2) + c * x1 - r) ** 2)
    term2 = s * (1.0 - t) * np.cos(x1)

    out = term1 + term2 + s
    return out


def test_hartmann6_impl():
    # Optimal _parameters
    x = [0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_hartmann6(0, params)
    assert np.allclose(loss, -3.32237)


def test_branin_impl():
    # Optimal parameter 1
    x = [-np.pi, 12.275]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin(0, params)
    print(loss)
    assert np.allclose(loss, 0.397887)

    # Optimal parameter 2
    x = [np.pi, 2.275]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin(0, params)
    assert np.allclose(loss, 0.397887)

    # Optimal parameter 3
    x = [9.42478, 2.475]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin(0, params)
    assert np.allclose(loss, 0.397887)


""" Keras Evaluators """


def evaluation_simple_keras_tf(session, worker_id, params):
    """

    Args:
        session (tf.Session | None):
        worker_id (int):
        params (data.OrderedDict):

    Returns:
        float
    """
    from keras import backend as K

    values = K.variable(list(params.values())[:3])
    metric = K.sum(values)
    metric = K.eval(metric)

    print('objective value =', metric)
    return metric


def evaluation_hartmann6_keras_tf(session, worker_id, params):
    """

    Args:
        session (tf.Session | None):
        worker_id (int):
        params (data.OrderedDict):

    Returns:

    """
    """ Code ported from https://www.sfu.ca/~ssurjano/Code/hart6scm.html
    Global Minimum = -3.32237
    """
    from keras import backend as K

    xx = K.variable(np.array([list(params.values())]).T)

    alpha = np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float32)

    A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], dtype=np.float32)

    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]],
                        dtype=np.float32)

    inner = K.sum(A * ((K.transpose(xx) - P) ** 2), axis=-1)
    inner = K.exp(-inner)
    outer = K.sum(alpha * inner)

    outer = K.eval(outer)

    return -outer


def evaluation_branin_keras_tf(session, worker_id, params):
    """

    Args:
        session (tf.Session | None):
        worker_id (int):
        params (data.OrderedDict):

    Returns:

    """
    """ Code ported from https://www.sfu.ca/~ssurjano/Code/braninm.html
    Global Minimum = -0.397887
    """
    from keras import backend as K

    xx = list(params.values())
    x1, x2 = xx[0], xx[1]

    a = 1.0
    b = 5.1 / (4 * (np.pi ** 2))
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    term1 = a * ((x2 - b * (x1 ** 2) + c * x1 - r) ** 2)
    term2 = s * (1.0 - t) * K.cos(x1)

    out = term1 + term2 + s

    out = K.eval(out)
    return out


def test_hartmann6_keras_tf_impl():
    # Optimal _parameters
    x = [0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_hartmann6_keras_tf(None, 0, params)
    assert np.allclose(loss, -3.32237)


def test_branin_keras_tf_impl():
    # Optimal parameter 1
    x = [-np.pi, 12.275]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin_keras_tf(None, 0, params)
    print(loss)
    assert np.allclose(loss, 0.397887)

    # Optimal parameter 2
    x = [np.pi, 2.275]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin_keras_tf(None, 0, params)
    assert np.allclose(loss, 0.397887)

    # Optimal parameter 3
    x = [9.42478, 2.475]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin_keras_tf(None, 0, params)
    assert np.allclose(loss, 0.397887)


@tf_optimizer_wrapper
def test_shac_simple():
    total_budget = 100
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.KerasSHAC(h, total_budget=total_budget, max_gpu_evaluators=1,
                            num_batches=batch_size, objective=objective)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 20
    assert shac.num_workers == 20
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 1
    shac.num_parallel_evaluators = 1

    print("Evaluating before training")
    np.random.seed(0)
    random_samples = shac.predict(num_batches=16, num_workers_per_batch=1)  # random sample predictions

    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    random_mean = np.mean(random_eval)

    print()

    # training
    shac.fit(evaluation_simple_keras_tf)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('shac/datasets/dataset.csv')
    assert os.path.exists('shac/classifiers/classifiers.pkl')

    print()
    print("Evaluating after training")
    np.random.seed(0)
    predictions = shac.predict(num_batches=16, num_workers_per_batch=1)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = engine.KerasSHAC(None, total_budget=total_budget, max_gpu_evaluators=0,
                             num_batches=10, objective=objective)

    shac2.restore_data()

    np.random.seed(0)
    predictions = shac.predict(num_batches=10, num_workers_per_batch=1)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # test no file found, yet no error
    shutil.rmtree('shac/')

    shac2.dataset = None
    shac2.classifiers = None
    shac2.restore_data()


@tf_optimizer_wrapper
def test_shac_simple_custom_basepath():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.KerasSHAC(h, total_budget=total_budget, max_gpu_evaluators=1,
                            num_batches=batch_size, objective=objective,
                            save_dir='custom')

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 10
    assert shac.num_workers == 10
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 1
    shac.num_parallel_evaluators = 1

    # training
    shac.fit(evaluation_simple_keras_tf)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('custom/datasets/dataset.csv')
    assert os.path.exists('custom/classifiers/classifiers.pkl')

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = engine.KerasSHAC(None, total_budget=total_budget, max_gpu_evaluators=0,
                             num_batches=10, objective=objective, save_dir='custom')

    shac2.restore_data()

    # test no file found, yet no error
    shutil.rmtree('custom/')

    shac2.dataset = None
    shac2.classifiers = None
    shac2.restore_data()


@tf_optimizer_wrapper
def test_shac_simple_early_stop():
    total_budget = 100
    batch_size = 20
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.KerasSHAC(h, total_budget=total_budget, max_gpu_evaluators=0,
                            num_batches=batch_size, objective=objective)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 5
    assert shac.num_workers == 5
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 1
    shac.num_parallel_evaluators = 1

    # training (with failure)
    shac.fit(evaluation_simple_keras_tf, early_stop=True, skip_cv_checks=True)
    assert len(shac.classifiers) == 0


if __name__ == '__main__':
    pytest.main([__file__])
