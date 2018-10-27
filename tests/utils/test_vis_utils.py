import os
import sys
import shutil
import six
import pytest
import warnings
import numpy as np

import matplotlib
matplotlib.use('Agg')

from pyshac.config import hyperparameters as hp, data
from pyshac.utils import vis_utils

# compatible with both Python 2 and 3
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


warnings.simplefilter('ignore')


# wrapper function to clean up saved files
def viz_wrapper(func):
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


def get_hyperparameter_list():
    h1 = hp.DiscreteHyperParameter('h1', [0, 1, 2])
    h2 = hp.DiscreteHyperParameter('h2', [3, 4, 5, 6])
    h3 = hp.UniformContinuousHyperParameter('h3', 1, 10)
    return [h1, h2, h3]


@viz_wrapper
def test_plot_dataset():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    # fit samples
    num_samples = 50
    samples = [h.sample() for _ in range(num_samples)]
    labels = [np.sum(sample) for sample in samples]

    for x, y in zip(samples, labels):
        dataset.add_sample(x, y)

    # Plot without saving
    vis_utils.plot_dataset(dataset, to_file=False)

    # Plot with saving
    vis_utils.plot_dataset(dataset)
    assert os.path.exists('dataset.png')

    os.remove('dataset.png')

    # Plot to directory with saving
    vis_utils.plot_dataset(dataset, to_file='temp/dataset.png')
    assert os.path.exists('temp/dataset.png')

    shutil.rmtree('temp')

    # None path
    with pytest.raises(FileNotFoundError):
        vis_utils.plot_dataset(None)

    # Wrong path
    with pytest.raises(FileNotFoundError):
        vis_utils.plot_dataset('random')

    # Empty dataset test
    with pytest.raises(ValueError):
        vis_utils.plot_dataset(data.Dataset())

    # Test empty title
    vis_utils.plot_dataset(dataset, title=None, to_file=None)

    # Test empty eval label
    vis_utils.plot_dataset(dataset, eval_label=None, to_file=None)

    # Test different degree of kernel
    vis_utils.plot_dataset(dataset, trend_deg=1, to_file=None)


if __name__ == '__main__':
    pytest.main([__file__])
