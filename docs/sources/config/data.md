# Datasets
----

The `Dataset` class is used to provide utilities for data management, such as adding training samples,
encoding and decoding data points for the training, serialization and restoration to continue training and so on.

A `Dataset` is an internal component of the algorithm being used, and will often not be interacted with directly.
It provides a convenient interface to the `HyperParameterList` that is used to wrap the hyper parameters, and its
serialization / recovery for multiple training runs.

## Class Information
----

<span style="float:right;">[[source]](https://github.com/titu1994/pyshac/blob/master/pyshac/config/data.py#L18)</span>
## [Dataset](#dataset)

```python
pyshac.config.data.Dataset(parameter_list=None)
```

Dataset manager for the engines.

Holds the samples and their associated evaluated values in a format
that can be serialized / restored as well as encoder / decoded for
training.

__Arguments:__

- **parameter_list (hp.HyperParameterList | list | None):** A python list
    of Hyper Parameters, or a HyperParameterList that has been built.
    Can also be None, if the parameters are to be assigned later.


---
## Dataset methods

### add_sample


```python
add_sample(parameters, value)
```



Adds a single row of data to the dataset.
Each row contains the hyper parameter configuration as well as its associated
evaluation measure.

__Arguments:__

- **parameters (list):** A list of hyper parameters that have been sampled
- **value (float):** The evaluation measure for the above sample.


---
### clear


```python
clear()
```



Removes all the data of the dataset.

---
### decode_dataset


```python
decode_dataset(X=None)
```



Decode the input samples such that discrete hyper parameters are mapped
to their original values and continuous valued hyper paramters are left alone.

__Arguments:__

- **X (np.ndarray | None):** The input list of encoded samples. Can be None,
    in which case it defaults to the internal samples, which are encoded
    and then decoded.

__Returns:__

 np.ndarray

---
### encode_dataset


```python
encode_dataset(X=None, Y=None, objective='max')
```



Encode the entire dataset such that discrete hyper parameters are mapped
to integer indices and continuous valued hyper paramters are left alone.

__Arguments__

- **X (list | np.ndarray | None):** The input list of samples. Can be None,
    in which case it defaults to the internal samples.
- **Y (list | np.ndarray | None):** The input list of evaluation measures.
    Can be None, in which case it defaults to the internal evaluation
    values.
- **objective (str):** Whether to maximize or minimize the
    value of the labels.

__Raises:__

- __ValueError__: If `objective` is not in [`max`, `min`]

__Returns:__

 A tuple of numpy arrays (np.ndarray, np.ndarray)

---
### get_dataset


```python
get_dataset()
```



Gets the entire dataset as a numpy array.

__Returns:__

(np.ndarray, np.ndarray)

---
### get_parameters


```python
get_parameters()
```



Gets the hyper parameter list manager

__Returns:__

HyperParameterList

---
### load_from_directory


```python
load_from_directory(dir='shac')
```



Static method to load the dataset from a directory.

__Arguments:__

- __dir__: The base directory where 'shac' directory is. It will build the path
    to the data and parameters itself.

__Raises:__

- __FileNotFoundError__: If the directory does not contain the data and parameters.

---
### prepare_parameter


```python
prepare_parameter(sample)
```



Wraps a hyper parameter sample list with the name of the
parameter in an OrderedDict.

__Arguments:__

- **sample (list):** A list of sampled hyper parameters

__Returns:__

 OrderedDict(str, int | float | str)

---
### restore_dataset


```python
restore_dataset()
```



Restores the entire dataset from a CSV file saved at the path provided by
`data_path`. Also loads the parameters (list of hyperparameters).

__Raises:__

- __FileNotFoundError__: If the dataset is not at the provided path.

---
### save_dataset


```python
save_dataset()
```



Serializes the entire dataset into a CSV file saved at the path
provide by `data_path`. Also saves the parameters (list of hyper parameters).

__Raises:__

- __ValueError__: If trying to save a dataset when its parameters have not been
    set.

---
### set_dataset


```python
set_dataset(X, Y)
```



Sets a numpy array as the dataset.

__Arguments:__

- **X (list | tuple | np.ndarray):** A numpy array or python list/tuple that contains
    the samples of the dataset.
- **Y (list | tuple | np.ndarray):** A numpy array or python list/tuple that contains
    the evaluations of the dataset.

---
### set_parameters


```python
set_parameters(parameters)
```



Sets the hyper parameter list manager

__Arguments:__

- **parameters (hp.HyperParameterList | list):** a Hyper Parameter List
    or a python list of Hyper Parameters.
