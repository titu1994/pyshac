import os
import shutil
import six
import pytest
import warnings
import numpy as np

from pyshac.config import hyperparameters as hp, data
from pyshac.utils import xgb_utils


# compatible with both Python 2 and 3
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


warnings.simplefilter('ignore')


# wrapper function to clean up saved files and be deterministic
def xgb_wrapper(func):
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        np.random.seed(0)
        output = func(*args, **kwargs)
        np.random.seed(None)

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
    h3 = hp.UniformContinuousHyperParameter('h3', 1, 10)
    return [h1, h2, h3]


@xgb_wrapper
def test_evaluate_single_sample():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    # models
    clfs = []

    # fit samples
    num_samples = 16
    for i in range(3):
        samples = [h.sample() for _ in range(num_samples)]
        labels = [np.sum(sample) for sample in samples]
        x, y = samples, labels
        x, y = dataset.encode_dataset(x, y)
        model = xgb_utils.train_single_model(x, y)
        clfs.append(model)

    # single sample test
    sample = h.sample()
    ex2, _ = dataset.encode_dataset([sample])

    assert ex2.shape == (1, 3)

    pred = xgb_utils.evaluate_models(ex2, clfs)
    assert pred.shape == (1,)


@xgb_wrapper
def test_evaluate_train_evaluate():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params, seed=0)

    dataset = data.Dataset(h)

    # models
    clfs = []

    # fit samples
    num_samples = 16
    for i in range(3):
        samples = [h.sample() for _ in range(num_samples)]
        labels = [np.sum(sample) for sample in samples]
        x, y = samples, labels
        x, y = dataset.encode_dataset(x, y)
        model = xgb_utils.train_single_model(x, y)
        clfs.append(model)

    # test samples
    num_samples = 100
    samples = [h.sample() for _ in range(num_samples)]
    ex2, _ = dataset.encode_dataset(samples, None)

    preds = xgb_utils.evaluate_models(ex2, clfs)
    count = np.sum(preds)

    print(count)
    assert preds.shape == (num_samples,)
    assert count > 0


@xgb_wrapper
def test_evaluate_train_evaluate_failure():
    params = [hp.DiscreteHyperParameter('h%d' % i, [0]) for i in range(3)]
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    # models
    clfs = []

    # fit samples
    num_samples = 16
    for i in range(3):
        samples = [h.sample() for _ in range(num_samples)]
        labels = [np.sum(sample) for sample in samples]
        x, y = samples, labels
        x, y = dataset.encode_dataset(x, y)
        model = xgb_utils.train_single_model(x, y)
        clfs.append(model)

    # test samples
    for model in clfs:
        assert model is None


@xgb_wrapper
def test_evaluate_train_evaluate_relax_check():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    # models
    clfs = []

    # fit samples
    num_samples = 16
    for i in range(3):
        samples = [h.sample() for _ in range(num_samples)]
        labels = [np.sum(sample) for sample in samples]
        x, y = samples, labels
        x, y = dataset.encode_dataset(x, y)
        model = xgb_utils.train_single_model(x, y)
        clfs.append(model)

    # test samples
    num_samples = 100
    samples = [h.sample() for _ in range(num_samples)]
    ex2, _ = dataset.encode_dataset(samples, None)

    for i in range(4):
        preds = xgb_utils.evaluate_models(ex2, clfs, relax_checks=i)
        count = np.sum(preds)

        print(count)
        assert preds.shape == (num_samples,)
        assert count > 0


@xgb_wrapper
def test_serialization_deserialization():
    basepath = 'shac'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    # models
    clfs = []

    # fit samples
    num_samples = 16
    for i in range(3):
        samples = [h.sample() for _ in range(num_samples)]
        labels = [np.sum(sample) for sample in samples]
        x, y = samples, labels
        x, y = dataset.encode_dataset(x, y)
        model = xgb_utils.train_single_model(x, y)
        clfs.append(model)

    xgb_utils.save_classifiers(clfs, basepath)
    assert os.path.exists(os.path.join(basepath, 'classifiers', 'classifiers.pkl'))

    models = xgb_utils.restore_classifiers(basepath)
    assert len(models) == len(clfs)

    with pytest.raises(FileNotFoundError):
        models = xgb_utils.restore_classifiers('none')


@xgb_wrapper
def test_serialization_deserialization_custom_basepath():
    basepath = 'custom'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h, basepath)

    # models
    clfs = []

    # fit samples
    num_samples = 16
    for i in range(3):
        samples = [h.sample() for _ in range(num_samples)]
        labels = [np.sum(sample) for sample in samples]
        x, y = samples, labels
        x, y = dataset.encode_dataset(x, y)
        model = xgb_utils.train_single_model(x, y)
        clfs.append(model)

    xgb_utils.save_classifiers(clfs, basepath)
    assert os.path.exists(os.path.join(basepath, 'classifiers', 'classifiers.pkl'))

    models = xgb_utils.restore_classifiers(basepath)
    assert len(models) == len(clfs)

    with pytest.raises(FileNotFoundError):
        models = xgb_utils.restore_classifiers('none')


if __name__ == '__main__':
    pytest.main([__file__])
