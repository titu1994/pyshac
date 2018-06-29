# Custom Hyper Parameters

There are 3 available hyper parameters made available by default:

- [DiscreteHyperParameter](config/hyperparameters.md#discretehyperparameter)
- [UniformContinuousHyperParameter](config/hyperparameters.md#uniformcontinuoushyperparameter)
- [NormalContinuousHyperParameter](config/hyperparameters.md#normalcontinuoushyperparameter)

However, if other parameters need to be added, this can be done as follows :

- Define a new class which extends `AbstractHyperParameter` or any of its subclasses (other than `HyperParameterList`).
- Register this new hyper parameter with the module so that it can be saved and restored.
- Use the new hyper parameter as normal.

## Define a new Hyper Parameter

A hyper parameter needs to define a few methods from the abstract class. They are :

- **`sample()`**: Sample a single value from the set of all possible values that this parameter can have.
- **`encode`**`: Encode a single value into an integer or floating point representation.
- **`decode`**`: Decode an integer or floating point representation to the original value that it represents.
- **`_cast`**`: Cast the value provided to the original data type that was provided in the constructor.
- **`get_config`**`: Prepare a config of the arguments to the constructor of this class, so that it can be restored.

!!!example ""
    The following is a mock example that replicates some of [UniformContinuousHyperParameter](config/hyperparameters.md#uniformcontinuoushyperparameter)

    We subclass `AbstractContinuousHyperParameter`, a subclass of `AbstractHyperParameter`, which defines useful functionality for
    hyper parameters that can have values in a range and can be sampled from a distribution.

    This subclass is useful, since it pre-defines several methods for us :

    - `encode`
    - `decode`
    - `_cast`


```python
import pyshac
from pyshac.config.hyperparameters import AbstractContinuousHyperParameter

class CustomUniformHP(AbstractContinuousHyperParameter):

    def __init__(self, name, min_value, max_value):
        super(CustomUniformHP, self).__init__(name, min_value, max_value)

    def sample(self):
        """
        Here, self._val1 and self._val2 are the private members of `AbstractContinuousHyperParameters`,
        which represent the range that this distribution can be sampled from.
        """
        value = np.random.uniform(self._val1, self._val2, size=1)[0]
        return value

    def get_config(self):
        """
        Creates the config of the class with all of its values.

        Note 1: The keys in this dict must match the argument names
            defined in the constructor of this class.

        Note 2: The `name` key is already assigned via the `base_config`,
            so there is no need to add that to the config again.
        """
        config = {
            'min_value': self.min_value,
            'max_value': self.max_value,
        }

        # add the base class config alongside this classes config
        base_config = super(UniformContinuousHyperParameter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

## Register the new hyper parameter

It takes just two lines to register this new hyper parameter. Simply use the `set_custom_parameter_class` method
defined in the `pyshac.config.hyperparameters` module to register a new class.

```python
from pyshac.config.hyperparemeters import set_custom_parameter_class

set_custom_parameter_class(CustomUniformHP)

```

## Use this new hyper parameter

After this registration, this new hyper parameter will be available to include in a list, training,
predicting new samples and saving and restoring state of the engine.

!!!warning "Remember to set the custom parameter **before** restoring the engine"
    Due to how the class is bound and re-created when being restored, it is important to
    set the custom parameter to the module before using `restore_data()` on any engine.
