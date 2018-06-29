# SHAC Engine
----

The core algorithm from the paper [Parallel Architecture and Hyperparameter Search via Successive Halving and Classification](https://arxiv.org/abs/1805.10255).

This engine provides an interface similar to Scikit-Learn, with two methods - `fit` and `predict` which training and parameter sampling respectively.

The data for the classifiers is generated via rejection sampling, using Joblib for parallel batch generation and evaluation.
It even allows for halted training - such that training can resume from the previous stage without any issue.

Two important user inputs to this class are :

- Evaluation function
- Hyper Parameter list

----

## **Note**

The engine will generate samples at each epoch of training, and supply them to the evaluation function which will also be
executed in parallel. Therefore, all stages of sample generation, training, evaluation must be thread safe.

While the training algorithm can manage thread safety of the generator and training modules, the evaluation module is
for the user to write.

Therefore, we provide wrappers such as `TensorflowSHAC`, `KerasSHAC` and `TorchSHAC` to allow safer evaluation. However, this comes
at significant cost of execution, as a new Tensorflow Graph and likewise, a new Keras Session must be built for each evaluator thread.

!!!info "Serial Evaluation"
    If parallel evaluation is not preferred, please refer the [Serial Evaluation](../serial-execution.md) page.

## Class Information
----

<span style="float:right;">[[source]](https://github.com/titu1994/pyshac/blob/master/pyshac/core/engine.py#L723)</span>
## [SHAC](#shac)

```python
pyshac.core.engine.SHAC(evaluation_function, hyperparameter_list, total_budget, num_batches=16, objective='max', max_classifiers=18)
```


The default and generic implementation of the SHAC algorithm. It is a wrapper over the
abstract class, and performs no additional maintenance over the evaluation function.

It is fastest engine, but assumes that the evaluation function is thread safe and
the system has sufficient memory to run several copies of the evaluation function
at the same time.

This engine is suitable for Numpy / PyTorch based work. Both numpy and PyTorch
dynamically allocate memory, and therefore, as long as the system has enough memory
to run multiple copies of the evaluation model, there is no additional memory management
to be done.

Still, for PyTorch, the number of evaluation processes should be carefully set,
so as not to exhaust all CPU / GPU memory during execution.

__Arguments:__

- **evaluation_function ((int, list) -> float):** The evaluation function is passed
    only the integer id (of the worker) and the sampled hyper parameters in
    an OrderedDict.  The evaluation function is expected to pass a python
    floating point number representing the evaluated value.
- **hyperparameter_list (hp.HyperParameterList | None):** A list of parameters
    (or a HyperParameterList) that are passed to define the search space.
    Can be None, so that it is loaded from disk instead.
- **total_budget (int):** `N`. Defines the total number of models to evaluate.
- **num_batches (int):** `M`. Defines the number of batches the work is distributed
    to. Must be set such that `total budget` is divisible by `batch size`.
- **objective (str):** Can be `max` or `min`. Whether to maximise
    the evaluation measure or minimize it.
- **max_classifiers (int):** Maximum number of classifiers that
    are trained. Default (18) is according to the paper.

__References:__

- [Parallel Architecture and Hyperparameter Search via Successive Halving and Classification](https://arxiv.org/abs/1805.10255)

__Raises:__

- __ValueError__: If `total budget` is not divisible by `batch size`.


---
## SHAC methods

### fit


```python
fit(skip_cv_checks=False, early_stop=False, relax_checks=False)
```



Generated batches of samples, trains `total_classifiers` number of XGBoost models
and evaluates each batch with the supplied function in parallel.

Allows manually changing the number of processes that are used to generate samples
or to evaluate them. While the defaults generally work well, further performance
gains can be had by trying different values according to the limits of the system.

```python
>>> eval = lambda id, params: np.exp(params['x'])
>>> shac = SHAC(eval, params, total_budget=100, num_batches=10)

>>> shac.num_parallel_generators = 20  # change the number of generator process
>>> shac.num_parallel_evaluators = 1  # change the number of evaluator processes
>>> shac.generator_backend = 'multiprocessing'  # change the backend for the generator
>>> shac.evaluator_backend = 'threading'  # change the backend of the evaluator
```

Has an adaptive behaviour based on what epoch it is on, since later epochs require
far more number of samples to generate a single batch of samples. When the epoch
number increases beyond 10, it doubles the number of generator processes.

This adaptivity can be removed by setting the parameter `limit_memory` to True.
```
>>> shac.limit_memory = True
```

__Arguments:__

- **skip_cv_checks (bool):** If set, will not perform 5 fold cross validation check
    on the models before adding them to the classifer list. Useful when the
    batch size is small.
- **early_stop (bool):** Stop running if fail to find a classifier that beats the
    last stage of evaluations.
- **relax_checks (bool):** If set, will allow samples who do not pass all of the
    checks from all classifiers. Can be useful when large number of models
    are present and remaining search space is not big enough to allow sample
    to pass through all checks.

---
### predict


```python
predict(num_batches=1, num_workers_per_batch=None, relax_checks=False, max_classfiers=None)
```



Using trained classifiers, sample the search space and predict which samples
can successfully pass through the cascade of classifiers.

When using a full cascade of 18 classifiers, a vast amount of time to sample
a single batch. It is recommended to save the model (done automatically during
training), restore a model with a smaller batch size and then use predict.

__Arguments:__

- **num_batches (int):** Number of batches, each of `num_batches` to be generated
- **num_workers_per_batch (int):** Determines how many parallel threads / processes
    are created to generate the batches. For small batches, it is best to use 1.
    If left as `None`, defaults to `num_parallel_generators`.
- **relax_checks (bool):** If set, will allow samples who do not pass all of the
    checks from all classifiers. Can be useful when large number of models
    are present and remaining search space is not big enough to allow sample
    to pass through all checks.
- **max_classfiers (int | None):** Number of classifiers to use for sampling.
    If set to None, will use all classifiers.

__Raises:__

- __ValueError__: If `max_classifiers` is larger than the number of available
    classifiers.

__Returns:__

batches of samples in the form of an OrderedDict

---
### save_data


```python
save_data()
```



Serialize the class objects by serializing the dataset and the trained models.

---
### restore_data


```python
restore_data()
```



Recover the serialized class objects by loading the dataset and the trained models
from the default save directories.

---
### set_num_parallel_generators


```python
set_num_parallel_generators(n)
```



Check and sets the number of parallel generators. If None, checks if the number
of workers exceeds the number of virtual cores. If it does, it warns about it
and sets the max parallel generators to be the number of cores.

__Arguments:__

- **n (int | None):** The number of parallel generators required.

---
### set_num_parallel_evaluators


```python
set_num_parallel_evaluators(n)
```



Check and sets the number of parallel evaluators. If None, checks if the number
of workers exceeds the number of virtual cores. If it does, it warns about it
and sets the max parallel generators to be the number of cores.

__Arguments:__

- **n (int | None):** The number of parallel evaluators required.

---
### parallel_evaluators


```python
parallel_evaluators()
```



Sets the evaluators to use the multiprocessing backend.

The user must take the responsibility of thread safety and memory management.

---
### concurrent_evaluators


```python
concurrent_evaluators()
```



Sets the evaluators to use the threading backend, and therefore
be locked by Python's GIL.

While technically still "parallel", it is in fact concurrent execution
rather than parallel execution of the evaluators.
