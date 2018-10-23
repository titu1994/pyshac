import os
import shutil
import six
import pytest
import warnings
import numpy as np

from pyshac.config import hyperparameters as hp, callbacks as cb
from pyshac.core import engine

warnings.simplefilter('ignore', DeprecationWarning)

# compatible with both Python 2 and 3
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


# wrapper function to clean up saved files and be deterministic
def cleanup_wrapper(func):
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        np.random.seed(0)
        output = func(*args, **kwargs)
        np.random.seed(None)

        # remove temporary files
        if os.path.exists('shac/'):
            shutil.rmtree('shac/')
        return output
    return wrapper


def create_mock_dataset():
    np.random.seed(0)

    with open('shac/mock.csv', 'w') as f:
        f.write('id,x,y,scores\n')
        f.flush()

        template = '%d,%0.4f,%0.4f,%0.5f\n'

        for i in range(1000):
            x = np.random.uniform(-1., 1.)
            y = np.random.normal(0, 5.)
            score = x ** 2 + y ** 3
            f.write(template % (i, x, y, score))
            f.flush()


def get_hyperparameter_list():
    h1 = hp.DiscreteHyperParameter('h1', [0, 1, 2])
    h2 = hp.DiscreteHyperParameter('h2', [3, 4, 5, 6])
    h3 = hp.UniformContinuousHyperParameter('h3', 1, 25)
    h4 = hp.DiscreteHyperParameter('h4', ['v1', 'v2'])
    return [h1, h2, h3, h4]


def evaluation_simple(worker_id, params):
    values = list(params.values())[:3]
    metric = np.sum(values)
    print('objective value =', metric)
    return metric


@cleanup_wrapper
def test_history_fit():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 10
    assert shac.num_workers == 10
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 2
    shac.num_parallel_evaluators = 2

    print("Evaluating before training")
    np.random.seed(0)

    # Create the callbacks
    history = cb.History()

    # training
    history = shac.fit(evaluation_simple, callbacks=[history])

    assert isinstance(history, cb.History)
    assert 'begin_run_index' in history.history
    assert 'model' in history.history
    assert 'parameters' in history.history
    assert 'evaluations' in history.history
    assert 'per_classifier_budget' in history.history
    assert 'generator_threads' in history.history
    assert 'device_ids' in history.history

    # Test passing in empty callback list

    # training
    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    history = shac.fit(evaluation_simple)

    assert isinstance(history, cb.History)
    assert 'begin_run_index' in history.history
    assert 'model' in history.history
    assert 'parameters' in history.history
    assert 'evaluations' in history.history
    assert 'per_classifier_budget' in history.history
    assert 'generator_threads' in history.history
    assert 'device_ids' in history.history


@cleanup_wrapper
def test_history_fit_dataset():
    total_budget = 1000
    batch_size = 5
    objective = 'max'

    params = [hp.UniformHP('x', -1., 1.), hp.NormalHP('y', 0., 5.)]
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    # create the mock dataset
    create_mock_dataset()

    print("Evaluating before training")
    np.random.seed(0)

    # Create the callbacks
    history = cb.History()

    # training
    history = shac.fit_dataset('shac/mock.csv', callbacks=[history])

    assert isinstance(history, cb.History)
    assert 'begin_run_index' in history.history
    assert 'model' in history.history
    assert 'parameters' in history.history
    assert 'evaluations' in history.history
    assert 'per_classifier_budget' in history.history

    # Test passing in empty callback list

    # training
    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    history = shac.fit_dataset('shac/mock.csv', presort=True)

    assert isinstance(history, cb.History)
    assert 'begin_run_index' in history.history
    assert 'model' in history.history
    assert 'parameters' in history.history
    assert 'evaluations' in history.history
    assert 'per_classifier_budget' in history.history


@cleanup_wrapper
def test_csvwriter_fit():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 10
    assert shac.num_workers == 10
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 2
    shac.num_parallel_evaluators = 2

    print("Evaluating before training")
    np.random.seed(0)

    # Create the callbacks
    callback = cb.CSVLogger('shac/logs.csv', append=True)

    # training
    shac.fit(evaluation_simple, callbacks=[callback])

    assert os.path.exists('shac/logs.csv')


@cleanup_wrapper
def test_csvwriter_fit_dataset():
    total_budget = 1000
    batch_size = 5
    objective = 'max'

    params = [hp.UniformHP('x', -1., 1.), hp.NormalHP('y', 0., 5.)]
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    # create the mock dataset
    create_mock_dataset()

    print("Evaluating before training")
    np.random.seed(0)

    # Create the callbacks
    callback = cb.CSVLogger('shac/logs.csv')

    # training
    shac.fit_dataset('shac/mock.csv', callbacks=[callback])

    assert os.path.exists('shac/logs.csv')


if __name__ == '__main__':
    pytest.main([__file__])
