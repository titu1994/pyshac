# Guide
----

PySHAC has a very simple interface, which follows similar semantics to Scikit-Learn. Simply importing as below grants the majority of the functionality of the library.

```python
import pyshac
```

There are three main processes that are followed when using PySHAC :

- **Declaration of hyper parameters**:
    -   Hyper parameters can be discrete or continuous, and if continuous, be sampled from different distributions.
    -   All hyper parameters must be defined before being passed to the engine.

- **Training of the models**:
    -   Once the parameters are defined, we create an instance of the engine.
    -   This engine is passed an evaluation function that the user must write, along with the hyper parameter declarations and other parameters as required (such as total budget and number of batches)
    -   The engine's `fit()` method is used to perform training of the classifiers.
    -   Training can be stopped at any time between epochs. All models are serialized at the end of each epoch.

- **Sampling the best hyper parameters**:
    -   Using the serialized models, we can sample the search space in order to obtain the best hyper parameters possible.
    -   This is done using the engine's `predict()` method.

## Important Note for Windows Platform

!!!warning
    The `fit` and `predict` methods **must be called only from inside of a `if __name__ == '__main__'` block.**

    This is a limitation of how Windows does not support forking, and so the engine definition and its methods must be called
    inside of a `__main__` block.

    It is simpler to put this code inside of a function, and simply call this function from the `__main__` block
    to have better readability of code.

## Declaration of Hyper Parameters
----

There are 3 available hyper parameters made available :

- [DiscreteHyperParameter](config/hyperparameters.md#discretehyperparameter)
- [UniformContinuousHyperParameter](config/hyperparameters.md#uniformcontinuoushyperparameter)
- [NormalContinuousHyperParameter](config/hyperparameters.md#normalcontinuoushyperparameter)

While these can declare most of the common hyper parameters, if there is a need for custom hyper parameters, then they
can very easily be added as shown in [Custom Hyper Parameters](custom-hyper-parameters.md).

----
### Naming hyper parameters

For the engine to work correctly, all hyper parameters must have **unique names** associated with them.

These names are the keyword arguments of the dictionary received by the evaluation function, so we can get the value
of the sample by this name inside the evaluation function.

----
### Declaring hyper parameters

Declaring hyper parameters is as easy as follows :

```python
import pyshac

# Discrete parameters
dice_rolls = pyshac.DiscreteHyperParameter('dice', values=[1, 2, 3, 4, 5, 6])
coin_flip = pyshac.DiscreteHyperParameter('coin', values=[0, 1])

# Continuous Parameters
classifier_threshold = pyshac.UniformContinuousHyperParameter('threshold', min_value=0.0, max_value=1.0)
noise = pyshac.NormalContinuousHyperParameter('noise', mean=0.0, std=1.0)

```

## Training of Models

There are multiple engines available for use, and their use case differs based on what work is being done.
For this guide, we are considering our task will be accomplished with python classifiers (XGBoost, Scikit-Learn etc)
which do not depend on GPU acceleration.

However, when working with Tensorflow, PyTorch or Keras models, it is advisable to use the appropriate engine.
More information can be found in [Managed Engines](managed.md).
----

### Setting up the engine

When setting up the SHAC engine, we need to define a few important parameters which will be used by the engine :

- **Evaluation Function**: This is a user defined function, that accepts 2 or more inputs as defined by the engine, and returns a python floating point value.
- **Hyper Parameter list**: A list of parameters that have been declared. This will constitute the search space.
- **Total budget**: The number of evaluations that will occur.
- **Number of batches**: The number of batches per epoch of evaluation.
- **Objective**: String value which can be either `max` or `min`. Defines whether the objective should be maximised or minimised.
- **Maximum number of classifiers**: As it suggests, decides the upper limit of how many classifiers can be trained. This is optional, and usually not required to specify.

!!!warning
    The total budget needs to be divisible by the number of batches, since all evaluations need to have the same number of samples.
    This eases the training of models as well, since we can be sure of how many samples are present after each epoch.

**Evaluation Function** receives 2 inputs :

- **Worker ID**: Integer id that can be left alone when executing only on CPU or used to determine the iteration number in the current epoch of evaluation.
- **Parameter OrderedDict**: An OrderedDict which contains the (name, value) pairs of the Parameters passed to the engine.
    -   Since it is an ordered dict, if only the values are required, `list(parameters.values())` can be used to get the list of values in the same order as when the Parameters were declared to the engine.
    -   These are the values of the sampled hyper parameters which have passed through the current cascade of models.

These parameters can be easily defined as :

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

While this looks like a lot, these few lines are in essence all that is required to define the search space,
the evaluation measure and the engine.

### Training the classifiers

Once the engine has been created, then it is simply a matter of calling `fit()` on the engine. This will create the
worker threads, generate the samples, test them if they pass all of the current classifiers, prepare the dataset for
that batch, test a cross validated classifier to see if it will train properly on the dataset, finally train a classifier
and then begin the next epoch.

```python

# assuming shac here is from the above code inside __main__
shac.fit()
```

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

## Sampling the best hyper parameters

Once the models have been trained by the engine, it is as simple as calling `predict()` to sample multiple batches of parameters.
As it is more efficient to sample several batches at once, `predict()` will return an a number of batches of sampled hyper paremeters.

```python

# sample a single batch of hyper parameters
parameter_samples = shac.predict()  # samples 1 batch, of batch size 10 (10 samples in total)

# sample more than one batch of hyper parameters
parameter_samples = shac.predict(10)  # samples 10 batches, each of batch size 10 (100 samples in total)
```


### Dealing with long sampling time

When using a very large number of classifiers (default of 18), it may take an enormous amount of time to sample a single
hyper parameter list, let alone a batch. Therefore, it is more efficient to reduce the checks if necessary by using a few parameters.

- **num_workers_per_batch**:
    -   When using a large batch size or large number of classifiers, it is advisable to have many parallel workers sampling batches at the same time to reduce the amount of time taken to get samples that pass through all classifiers.
    -   When using a small batch size or small number of classifiers, it is advisable to reduce the time taken to sample each batch by reducing the number of parallel workers.
    -   If left as `None`, will determine the value that was used during training.

- **relax_checks**: Same as for training, relaxes the checks during evaluation to sample parameters faster.
- **max_classfiers**: Another way to reduce the time taken to sample parameters is to simply reduce the number of classifiers used to perform the check. This will use the first `K` classifier stages to perform the checks.

```python

# batch size is 16, so using 16 processes to sample in parallel is more efficient than the default memory conservative count.
parameter_samples = shac.predict(10, num_workers_per_batch=16, relax_checks=True, max_classfiers=16)
```

## Continuing Training

As mentioned at the beginning of this guide, training can be stopped at any time after an epoch has finished.
After this, training run can be resumed from the last epoch successfully trained with only slight changes to how we
define and use the engine before using `fit()`.

There are a few steps, and a few important cautions that must be taken when continuing training :

- Create an instance of the engine in the same location as where the earlier model was trained.
    - It must be provided the same evaluation measure as before.
    - It is optional to provide the hyper parameter list. It is not required as it will be replaced by the parameters in the file.
    - If training further, **it is essential to use the same `total_budget`, `batch_size` and `objective` as before.**
    - If only evaluating the trained classifiers, using any value of `total_budget` is allowed (though the earlier value is better for consistency), however it must be divisible by `batch_size` and the `objective` must be same as before.
    - When restoring data, it must find the `shac` folder and all of the sub-folders in the same directory level.
- Call `restore_data()` on the engine. It will print out the number of samples and the number of classifiers it was able to find and load.
- Call `fit()` or `predict()` as before.

In doing so, the engine instance will be restored to the exact state as in the previous training session, and the new classifiers can be trained as normal.

Example :

```python

# Reduced batch_size allows faster sampling
# None for the hyper parameter list as it will be loaded in the restoration step.
new_shac = pyshac.SHAC(squared_error_loss, None, total_budget, batch_size=5, objective=objective)

# restore the engine
new_shac.restore_data()

# predict or train like before
new_shac.fit()

or

samples = new_shac.predict()
```