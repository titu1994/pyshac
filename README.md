# PySHAC : A Python Library for `Sequential Halving and Classification` Algorithm

[![Build Status](https://travis-ci.com/titu1994/pyshac.svg?token=Kpa1jkwxwsMcnGRBC8S2&branch=master)](https://travis-ci.com/titu1994/pyshac)
[![Coverage Status](https://coveralls.io/repos/github/titu1994/pyshac/badge.svg?branch=master)](https://coveralls.io/github/titu1994/pyshac?branch=master)

PySHAC is a python library to use the Sequential Halving and Classification algorithm from the paper
[Parallel Architecture and Hyperparameter Search via Successive Halving and Classification](https://arxiv.org/abs/1805.10255) with ease.

Note : This library is not affiliated with Google.

## Installation

This library is available for Python 2.7 and 3.4+ via pip for Windows, MacOSX and Linux.

```python
pip install pyshac
```

To install the master branch of this library :

```
git clone https://github.com/titu1994/pyshac.git
cd pyshac
python setup.py install
```

To install the requirements before installing the library :

```
pip install -r "requirements.txt"
```

To build the docs, additional packages must be installed :
```
pip install -r "doc_requirements.txt"
```

## Documentation

Stable build documentation can be found at [PySHAC Documentation](http://titu1994.github.io/pyshac/).

It contains a User Guide, as well as explanation of the different engines that can be used with PySHAC.

## Getting started with PySHAC

### First, build the set of hyper parameters. The three main HyperParameter classes are :

- DiscreteHyperParameter
- UniformContinuousHyperParameter
- NormalContinuousHyperParameter

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

- **Evaluation Function**: This is a user defined function, that accepts 2 or more inputs as defined by the engine, and returns a python floating point value.
- **Hyper Parameter list**: A list of parameters that have been declared. This will constitute the search space.
- **Total budget**: The number of evaluations that will occur.
- **Number of batches**: The number of samples per batch of evaluation.
- **Objective**: String value which can be either `max` or `min`. Defines whether the objective should be maximised or minimised.
- **Maximum number of classifiers**: As it suggests, decides the upper limit of how many classifiers can be trained. This is optional, and usually not required to specify.

The **Evaluation Function** receives 2 inputs :

- **Worker ID**: Integer id that can be left alone when executing only on CPU or used to determine the iteration number in the current epoch of evaluation.
- **Parameter OrderedDict**: An OrderedDict which contains the (name, value) pairs of the Parameters passed to the engine.
    -   Since it is an ordered dict, if only the values are required, `list(parameters.values())` can be used to get the list of values in the same order as when the Parameters were declared to the engine.
    -   These are the values of the sampled hyper parameters which have passed through the current cascade of models.

```python

import numpy as np
import pyshac

# define the evaluation function
def squared_error_loss(id, parameters):
    x = parameters['x']
    y = parameters['y']
    y_sample = 2 * x - y

    # assume best values of x and y and 2 and 0 respectively
    y_true = 4.

    return np.square(y_sample - y_true)

if __name__ == '__main__':  # this is required for Windows ; not for Unix or Linux

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

    shac = pyshac.SHAC(squared_error_loss, parameters, total_budget, num_batches, objective)
```


### Training the classifiers

A single call to `shac.fit()` will begin training the classifiers.

There are a few cases to consider:

- There can be cases where the search space is not large enough to train the maximum number of classifier (usually 18).
- There may be instances where we want to allow some relaxations of the constraint that the next batch must pass through all
of the previous classifiers. This allows classifiers to train on the same search space repeatedly rather than divide the search space.

In these cases, we can utilize a few arguments to allow the training behaviour to better adapt to these circumstances.
These parameters are :

- **skip_cv_checks**: As it suggests, if the number of samples per batch is too small, it is preferable to skip the cross validation check, as most classifiers will not pass them.
- **early_stopping**: Determines whether training should halt as soon as an epoch of failed learning occurs. This is useful when evaluations are very costly.
- **relax_checks**: This will instead relax the constrain of having the sample pass through all classifiers to having the classifier past through most of the classifiers. In doing so, more samples can be obtained for the same search space.

```python

# `early stopping` default is False, and it is preferred not to use it when using `relax checks`
shac.fit(skip_cv_checks=True, early_stopping=False, relax_checks=True)
```

### Sampling the best hyper parameters

Once the models have been trained by the engine, it is as simple as calling `predict()` to sample multiple batches of parameters.
As it is more efficient to sample several batches at once, `predict()` will return an a number of batches of sampled hyper paremeters.

```python

# sample a single batch of hyper parameters
parameter_samples = shac.predict()  # samples 1 batch, of batch size 10 (10 samples in total)

# sample more than one batch of hyper parameters
parameter_samples = shac.predict(10)  # samples 10 batches, each of batch size 10 (100 samples in total)
```

## Examples

Examples based on the `Branin` and `Hartmann6` problems can be found in the [Examples folder](https://github.com/titu1994/pyshac/tree/master/examples).

An example of how to use the `TensorflowSHAC` engine is provided [in the example foldes as well](https://github.com/titu1994/pyshac/tree/master/examples/tensorflow).

----
