# PySHAC : A Python Library for `Sequential Halving and Classification` Algorithm

[![Build Status](https://travis-ci.org/titu1994/pyshac.svg?branch=master)](https://travis-ci.org/titu1994/pyshac)
[![codecov](https://codecov.io/gh/titu1994/pyshac/branch/master/graph/badge.svg)](https://codecov.io/gh/titu1994/pyshac)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/titu1994/pyshac/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyshac.svg)](https://pypi.org/project/pyshac/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyshac.svg)](https://pypi.org/project/pyshac/)
[![Downloads](https://pepy.tech/badge/pyshac)](https://pypi.org/project/pyshac/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyshac.svg)](https://pypi.org/project/pyshac/)
----

PySHAC is a python library to use the Sequential Halving and Classification algorithm from the paper
[Parallel Architecture and Hyperparameter Search via Successive Halving and Classification](https://arxiv.org/abs/1805.10255) with ease.

Note : This library is not affiliated with Google.

## Documentation

Stable build documentation can be found at [PySHAC Documentation](http://titu1994.github.io/pyshac/).

It contains a User Guide, as well as explanation of the different engines that can be used with PySHAC.


|   Topic     |  Link  |
|:-------------|:--------:|
| Installation | http://titu1994.github.io/pyshac/install/   |
| User Guide   |   http://titu1994.github.io/pyshac/guide/  |
| Managed Engines   |   http://titu1994.github.io/pyshac/managed/  |
| Custom Hyper Parameters   |   http://titu1994.github.io/pyshac/custom-hyper-parameters/  |
| Serial Evaluation   |   http://titu1994.github.io/pyshac/serial-execution/  |
| External Dataset Training   |   http://titu1994.github.io/pyshac/external-dataset-training/  |
| Callbacks   |   http://titu1994.github.io/pyshac/callbacks/  |


## Installation

This library is available for Python 2.7 and 3.4+ via pip for Windows, MacOSX and Linux.

```python
pip install pyshac
```

To install the master branch of this library :

```
git clone https://github.com/titu1994/pyshac.git
cd pyshac
pip install .

or pip install .[tests]  # to also include dependencies necessary for testing
```

To install the requirements before installing the library :

```
pip install -r "requirements.txt"
```

To build the docs, additional packages must be installed :
```
pip install -r "doc_requirements.txt"
```

## Getting started with PySHAC

### First, build the set of hyper parameters. The three main HyperParameter classes are :

- DiscreteHyperParameter
- UniformContinuousHyperParameter
- NormalContinuousHyperParameter

There are also 3 additional hyper parameters, which are useful when a parameter needs to be sampled multiple times
for each evaluation :

- MultiDiscreteHyperParameter
- MultiUniformContinuousHyperParameter
- MultiNormalContinuousHyperParameter

These multi parameters have an additional argument `sample_count` which can be used to sample multiple times
per step.

**Note**: The values will be concatenated linearly, so each multi parameter will have a list of values
returned in the resultant OrderedDict. If you wish to flatten the entire search space, you can
use `pyshac.flatten_parameters` on this OrderedDict.

```python
import pyshac

# Discrete parameters
dice_rolls = pyshac.DiscreteHyperParameter('dice', values=[1, 2, 3, 4, 5, 6])
coin_flip = pyshac.DiscreteHyperParameter('coin', values=[0, 1])

# Continuous Parameters
classifier_threshold = pyshac.UniformContinuousHyperParameter('threshold', min_value=0.0, max_value=1.0)
noise = pyshac.NormalContinuousHyperParameter('noise', mean=0.0, std=1.0)

```

### Setup the engine

When setting up the SHAC engine, we need to define a few important parameters which will be used by the engine :

- **Hyper Parameter list**: A list of parameters that have been declared. This will constitute the search space.
- **Total budget**: The number of evaluations that will occur.
- **Number of batches**: The number of samples per batch of evaluation.
- **Objective**: String value which can be either `max` or `min`. Defines whether the objective should be maximised or minimised.
- **Maximum number of classifiers**: As it suggests, decides the upper limit of how many classifiers can be trained. This is optional, and usually not required to specify.

```python

import numpy as np
import pyshac

# define the parameters
param_x = pyshac.UniformContinuousHyperParameter('x', -5.0, 5.0)
param_y = pyshac.UniformContinuousHyperParameter('y', -2.0, 2.0)

parameters = [param_x, param_y]

# define the total budget as 100 evaluations
total_budget = 100  # 100 evaluations at maximum

# define the number of batches
num_batches = 10  # 10 samples per batch

# define the objective
objective = 'min'  # minimize the squared loss

shac = pyshac.SHAC(parameters, total_budget, num_batches, objective)
```


### Training the classifiers

To train a classifier, the user must define an Evaluation function. This is a user defined function,
that accepts 2 or more inputs as defined by the engine, and returns a python floating point value.

The **Evaluation Function** receives at least 2 inputs :

- **Worker ID**: Integer id that can be left alone when executing only on CPU or used to determine the iteration number in the current epoch of evaluation.
- **Parameter OrderedDict**: An OrderedDict which contains the (name, value) pairs of the Parameters passed to the engine.
    -   Since it is an ordered dict, if only the values are required, `list(parameters.values())` can be used to get the list of values in the same order as when the Parameters were declared to the engine.
    -   These are the values of the sampled hyper parameters which have passed through the current cascade of models.

An example of a defined evaluation function :

```python
# define the evaluation function
def squared_error_loss(id, parameters):
    x = parameters['x']
    y = parameters['y']
    y_sample = 2 * x - y

    # assume best values of x and y and 2 and 0 respectively
    y_true = 4.

    return np.square(y_sample - y_true)
```

A single call to `shac.fit()` will begin training the classifiers.

There are a few cases to consider:

- There can be cases where the search space is not large enough to train the maximum number of classifier (usually 18).
- There may be instances where we want to allow some relaxations of the constraint that the next batch must pass through all
of the previous classifiers. This allows classifiers to train on the same search space repeatedly rather than divide the search space.

In these cases, we can utilize a few additional parameters to allow the training behaviour to better adapt to these circumstances.
These parameters are :

- **skip_cv_checks**: As it suggests, if the number of samples per batch is too small, it is preferable to skip the cross validation check, as most classifiers will not pass them.
- **early_stop**: Determines whether training should halt as soon as an epoch of failed learning occurs. This is useful when evaluations are very costly.
- **relax_checks**: This will instead relax the constrain of having the sample pass through all classifiers to having the classifier past through most of the classifiers. In doing so, more samples can be obtained for the same search space.

```python

# `early stopping` default is False, and it is preferred not to use it when using `relax checks`
shac.fit(squared_error_loss, skip_cv_checks=True, early_stop=False, relax_checks=True)
```

## Sampling the best hyper parameters

Once the models have been trained by the engine, it is as simple as calling `predict()` to sample multiple samples or batches of parameters.

Samples can be obtained in a per instance or per batch (or even a combination) using the two parameters - `num_samples` and `num_batches`.

```python

# sample a single instance of hyper parameters
parameter_samples = shac.predict()  # Gets 1 sample.

# sample multiple instances of hyper parameters
parameter_samples = shac.predict(10)  # Gets 10 samples.

# sample a batch of hyper parameters
parameter_samples = shac.predict(num_batches=5)  # samples 5 batches, each containing 10 samples.

# sample multiple batches and a few additional instances of hyper parameters
parameter_samples = shac.predict(5, 5)  # samples 5 batches (each containing 10 samples) and an additional 5 samples.
```

## Examples

Examples based on the `Branin` and `Hartmann6` problems can be found in the [Examples folder](https://github.com/titu1994/pyshac/tree/master/examples).

An example of how to use the `TensorflowSHAC` engine is provided [in the example foldes as well](https://github.com/titu1994/pyshac/tree/master/examples/tensorflow).

Comparison scripts of basic optimization, `Branin` and `Hartmann6` using Tensorflow Eager 1.8 are provided in the respective folders.

### Evaluation of Branin

Brannin to close to the true optima as described in the paper.

<img src='https://github.com/titu1994/pyshac/blob/master/examples/branin/dataset.png?raw=true' height=100% width=50%>

### Evaluation of Hardmann 6

Hartmann 6 was a much harder dataset, and results are worse than Random Search 2x and the one from the paper. Perhaps it was due to a bad run, and may be fixed with larger budget for training.

<img src='https://github.com/titu1994/pyshac/blob/master/examples/hartmann6/dataset.png?raw=true' height=100% width=50%>

### Evaluation of Simple Optimization Objective

The task is to sample two parameters `x` and `y`, such that `z = 2 * x - y` and we want `z` to approach the value of 4. We utilize MSE 
as the metric between z and the optimal value.

<img src='https://github.com/titu1994/pyshac/blob/master/examples/basic_opt/dataset.png?raw=true' height=100% width=50%>

### Evaluation of Hyper Parameter Optimization

The task is to sample hyper parameters which provide high accuracy values using TensorflowSHAC engine.

<img src='https://github.com/titu1994/pyshac/blob/master/examples/tensorflow/dataset.png?raw=true' height=100% width=50%>

----
