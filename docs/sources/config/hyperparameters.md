# Hyper Parameters
----

There are 3 primary hyper parameters :

- DiscreteHyperParameter (DiscreteHP)
- UniformContinuousHyperParameter (UniformHP)
- NormalContinuousHyperParameter (NormalHP)

As their names suggest, each hyper paramter takes specific parameters or value ranges.

- Discrete Hyper Parmeters takes lists of any python data type that can be serialized.
    -   This includes numpy values, and even numpy arrays

- Continuous Hyper Parameters take ranges of values
    -   Uniform Continuous Hyper Parameters will sample uniformly between the min and max values
    -   Normal Continuous Hyper Parameters will sample from the normal distribution with the provided mean and standard deviation

A special composite hyper parameter, called `HyperParameterList` is used to provide a convenient interface to several
hyper parameters at once, and is used internally by the training algorithm.

## Class Information
----

<span style="float:right;">[[source]](https://github.com/titu1994/pyshac/blob/master/pyshac/config/hyperparameters.py#L16)</span>
## [AbstractHyperParameter](#abstracthyperparameter)

```python
pyshac.config.hyperparameters.AbstractHyperParameter(name, values)
```


Abstract Hyper Parameter that defines the methods that all hyperparameters
need to supply

__Arguments:__

- **name (str):** Name of the hyper parameter
- **values (List, None):** A list of values (must all be pickle-able and hashable)
    values or None. If None, it is assumed to be a continuous value generator.

__Raises:__

- __ValueError__: If the `name` is not specified.


---
## AbstractHyperParameter methods

### decode


```python
decode(x)
```



Abstract method that defines how the parameter is decoded so
that the model can be properly trained.

__Arguments:__

- **x (int | float):** an encoded value that needs to be decoded.

__Raises:__

- __NotImplementedError__: Must be overridden by the subclass.

__Returns:__

a decoded value for the encoded input `x`.

---
### encode


```python
encode(x)
```



Abstract method that defines how the parameter is encoded
so that the model can properly be trained.

__Arguments:__

- **x (int | float | str):** a single value that needs to be encoded.

__Raises:__

- __NotImplementedError__: Must be overridden by the subclass.

__Returns:__

an encoded representation of the value of `x`.

---
### get_config


```python
get_config()
```



Creates the config of the class with all of its values.

__Returns:__

a dictionary with the config of the class.

---
### load_from_config


```python
load_from_config(config)
```

---
### sample


```python
sample()
```



Abstract method that defines how parameters are sampled.

__Raises:__

- __NotImplementedError__: Must be overridden by the subclass.

__Returns:__

a singular value sampled from possible values.

----

<span style="float:right;">[[source]](https://github.com/titu1994/pyshac/blob/master/pyshac/config/hyperparameters.py#L231)</span>
## [AbstractContinuousHyperParameter](#abstractcontinuoushyperparameter)

```python
pyshac.config.hyperparameters.AbstractContinuousHyperParameter(name, val1, val2, log_encode=False)
```


An abstract hyper parameter that represents a parameter that can take a range
of values from a certain distribution.

__Arguments:__

- **name (str):** Name of the parameter.
- **val1 (float):** A symbolic value that is used by subclasses.
- **val2 (float):** A symbolic value that is used by subclasses.
- **log_encode (bool):** Determines whether the encoding must be in natural
    log-space or not.

__Raises:__

- __NotImplementedError__: If `sample()` is called.


---
## AbstractContinuousHyperParameter methods

### decode


```python
decode(x)
```



Decodes the floating point value into normal space if `log_space` was set in
the constructor, else returns its original value.

__Arguments:__

- **x (float):** a single encoded sample.

__Returns:__

 float.

---
### encode


```python
encode(x)
```



Encodes the floating point value into log space if `log_space` was set in
the constructor, else returns its original value.

__Arguments:__

- **x (float):** a single sample.

__Returns:__

 float.

---
### get_config


```python
get_config()
```



Creates the config of the class with all of its values.

__Returns:__

a dictionary with the config of the class.

---
### load_from_config


```python
load_from_config(config)
```

---
### sample


```python
sample()
```



Abstract method that must be redefined by base classes.

__Returns:__

a float value.

----

<span style="float:right;">[[source]](https://github.com/titu1994/pyshac/blob/master/pyshac/config/hyperparameters.py#L141)</span>
## [DiscreteHyperParameter](#discretehyperparameter)

```python
pyshac.config.hyperparameters.DiscreteHyperParameter(name, values)
```


Discrete Hyper Parameter that defines a set of discrete values that it can take.

__Arguments:__

- **name (str):** Name of the hyper parameter.
- **values (list):** A list of values (must all be pickle-able and hashable)
    values or None.

__Raises:__

- __ValueError__: If the `name` is not specified.
.

__Raises:__

- __ValueError__: If the `name` is not specified.


---
## DiscreteHyperParameter methods

### decode


```python
decode(x)
```



Decodes a single encoded integer into its original value.

__Args:__

- **x (int):** an integer encoded value.

__Returns:__

 (int | float | str) representing the actual decoded value.

---
### encode


```python
encode(x)
```



Encodes a single value into an integer index.

__Arguments:__

- **x (int | float | str):** A value sampled from its possible values.

__Returns:__

 int value representing its encoded index.

---
### get_config


```python
get_config()
```



Creates the config of the class with all of its values.

__Returns:__

a dictionary with the config of the class.

---
### load_from_config


```python
load_from_config(config)
```

---
### sample


```python
sample()
```



Samples a single value from its set of discrete values.

__Returns:__

a single value from its list of possible values.

----

<span style="float:right;">[[source]](https://github.com/titu1994/pyshac/blob/master/pyshac/config/hyperparameters.py#L335)</span>
## [UniformContinuousHyperParameter](#uniformcontinuoushyperparameter)

```python
pyshac.config.hyperparameters.UniformContinuousHyperParameter(name, min_value, max_value, log_encode=False)
```


A hyper parameter that represents a parameter that can take a range
of values from a uniform distribution.

__Arguments:__

- **name (str):** Name of the parameter.
- **min_value (float):** The minimum value (inclusive) that the uniform
    distribution can take.
- **max_value (float):** The maximum value (exclusive) that the uniform
    distribution can take.
- **log_encode (bool):** Determines whether the encoding must be in natural
    log-space or not.


---
## UniformContinuousHyperParameter methods

### decode


```python
decode(x)
```



Decodes the floating point value into normal space if `log_space` was set in
the constructor, else returns its original value.

__Arguments:__

- **x (float):** a single encoded sample.

__Returns:__

 float.

---
### encode


```python
encode(x)
```



Encodes the floating point value into log space if `log_space` was set in
the constructor, else returns its original value.

__Arguments:__

- **x (float):** a single sample.

__Returns:__

 float.

---
### get_config


```python
get_config()
```



Creates the config of the class with all of its values.

__Returns:__

a dictionary with the config of the class.

---
### load_from_config


```python
load_from_config(config)
```

---
### sample


```python
sample()
```



Samples uniformly from the range [min_value, max_value).

__Returns:__

float.

----

<span style="float:right;">[[source]](https://github.com/titu1994/pyshac/blob/master/pyshac/config/hyperparameters.py#L388)</span>
## [NormalContinuousHyperParameter](#normalcontinuoushyperparameter)

```python
pyshac.config.hyperparameters.NormalContinuousHyperParameter(name, mean, std)
```


A hyper parameter that represents a parameter that can take a range
of values from a normal distribution.

__Arguments:__

- **name (str):** Name of the parameter.
- **mean (float):** The mean of the normal distribution.
- **std (float):** The standard deviation of the normal distribution.


---
## NormalContinuousHyperParameter methods

### decode


```python
decode(x)
```



Decodes the floating point value into normal space if `log_space` was set in
the constructor, else returns its original value.

__Arguments:__

- **x (float):** a single encoded sample.

__Returns:__

 float.

---
### encode


```python
encode(x)
```



Encodes the floating point value into log space if `log_space` was set in
the constructor, else returns its original value.

__Arguments:__

- **x (float):** a single sample.

__Returns:__

 float.

---
### get_config


```python
get_config()
```



Creates the config of the class with all of its values.

__Returns:__

a dictionary with the config of the class.

---
### load_from_config


```python
load_from_config(config)
```

---
### sample


```python
sample()
```



Samples from the normal distribution with a mean and standard deviation
as specified in the constructor.

__Returns:__

float.

----

<span style="float:right;">[[source]](https://github.com/titu1994/pyshac/blob/master/pyshac/config/hyperparameters.py#L436)</span>
## [HyperParameterList](#hyperparameterlist)

```python
pyshac.config.hyperparameters.HyperParameterList(hyper_parameter_list=None)
```


A composite hyper parameter, that encloses a list of hyper parameters
(either discrete or continuous) and provides utility methods for efficient
handling by the engine.

__Arguments:__

- **hyper_parameter_list (list(AnstractHyperParameter) | None):** A list of
    hyper parameters or None (which initializes this with 0 elements).


---
## HyperParameterList methods

### add_hyper_parameter


```python
add_hyper_parameter(parameter)
```



Adds a single hyper parameter (discrete or continuous) to the list
of hyper parameters managed by this HyperParameterList.

__Arguments:__

- **parameter (AbstractHyperParameter):** a subclass of AbstractHyperParameter,
    which will be embedded into this composite class.

__Raises:__

- __ValueError__: If the passed parameter is `None`, or the name already
    exists in the list of managed parameters.

---
### remove_hyper_parameter


```python
remove_hyper_parameter(parameter)
```



Removes a single hyper parameter (discrete or continuous) from the list
of hyper parameters managed by this HyperParameterList.

__Arguments:__

- **parameter (AbstractHyperParameter, str):** A string name or a subclass
    of AbstractHyperParameter which needs to be removed.

__Raises:__

- __ValueError__: If the passed parameter is `None`.

---
### sample


```python
sample()
```



Samples all of its component parameters and returns a list of the samples.

__Returns:__

 list of sampled parameters.

---
### encode


```python
encode(x)
```



Encodes a list of sampled hyper parameters.

__Arguments:__

- **x (list | np.ndarray):** A python list or numpy array of samples
    from the list of hyper parameters.

__Raises:__

- __ValueError__: If a numpy array of more than 1 dimension is provided.

__Returns:__

 ndarray(float).

---
### decode


```python
decode(x)
```



Decodes a list of sampled hyper parameters.

__Arguments:__

x (list(int | float)): a list of encoded integer or floating point
    values that are to be decoded.

__Returns:__

 list of decoded samples.

---
### get_config


```python
get_config()
```



Creates the config of the class with all of its values.

__Returns:__

an ordered dictionary with the config of the class.

---
### load_from_config


```python
load_from_config(config)
```

---
### get_parameter_names


```python
get_parameter_names()
```



Gets a list of all the parameter names managed by this class.

__Returns:__

a list(str) with the names of the parameters.

----

### get_parameter


```python
get_parameter(name)
```



Utility method to get the hyper parameter class by its name.

__Arguments:__

- **name (str):** Name of the class or its alias.

__Raises:__

- __ValueError__: If the class with the provided name does not exists in
    the set of available parameters.

__Returns:__

The hyper parameter class.

----

### set_custom_parameter_class


```python
set_custom_parameter_class(cls)
```



Utility function to dynamically add a custom hyper parameter
to the set of available hyper parameters.

__Arguments:__

- **cls (cls):** A class which extends `AbstractHyperParameter` in some way
    and implements the abstract methods.
