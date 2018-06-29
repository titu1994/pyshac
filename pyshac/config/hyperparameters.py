from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import sys
import inspect
import numpy as np
import uuid


# compatible with Python 2 *and* 3:
ABC = ABCMeta('ABC', (object,), {'__slots__': ()})


_CUSTOM_PARAMETERS = OrderedDict()


class AbstractHyperParameter(ABC):
    """
    Abstract Hyper Parameter that defines the methods that all hyperparameters
    need to supply

    # Arguments:
        name (str): Name of the hyper parameter
        values (List, None): A list of values (must all be pickle-able and hashable)
            values or None. If None, it is assumed to be a continuous value generator.

    # Raises:
        ValueError: If the `name` is not specified.
    """
    def __init__(self, name, values):

        if name is None:
            raise ValueError("`name` of the hyperparameter cannot be `None`")

        self.name = name
        self.num_choices = len(values) if values is not None else 0
        self.param2id = OrderedDict()
        self.id2param = OrderedDict()
        self.param2type = OrderedDict()

    @abstractmethod
    def sample(self):
        """
        Abstract method that defines how parameters are sampled.

        # Raises:
            NotImplementedError: Must be overridden by the subclass.

        # Returns:
            a singular value sampled from possible values.
        """
        raise NotImplementedError()

    @abstractmethod
    def encode(self, x):
        """
        Abstract method that defines how the parameter is encoded
        so that the model can properly be trained.

        # Arguments:
            x (int | float | str): a single value that needs to be encoded.

        # Raises:
            NotImplementedError: Must be overridden by the subclass.

        # Returns:
            an encoded representation of the value of `x`.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode(self, x):
        """
        Abstract method that defines how the parameter is decoded so
        that the model can be properly trained.

        # Arguments:
            x (int | float): an encoded value that needs to be decoded.

        # Raises:
            NotImplementedError: Must be overridden by the subclass.

        # Returns:
            a decoded value for the encoded input `x`.
        """
        raise NotImplementedError()

    @abstractmethod
    def _cast(self, x):
        """
        Casts the given value to its original data type.

        # Arguments:
            x (int | float | str): Input sample that will be cast to the
                correct data type.

        # Returns:
            the sample cast to the correct data type.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_config(self):
        """
        Creates the config of the class with all of its values.

        # Returns:
            a dictionary with the config of the class.
        """
        config = {
            'name': self.name,
        }
        return config

    @classmethod
    def load_from_config(cls, config):
        return cls(**config)

    def _build_maps(self, values):
        """
        Prepares a pair of dictionaries to manage the values provided.

        # Arguments:
            values (List, None): A list of values that are embedded into
                a pair of dictionaries. All values must be pickle-able and hashable.
        """
        if values is not None:
            for i, v in enumerate(values):
                self.param2id[v] = i
                self.id2param[i] = v

                # prepare a type map from string to its type, for fast checks
                self.param2type[v] = type(v)
                self.param2type[str(v)] = type(v)

    def __repr__(self):
        s = self.name + " : "
        vals = list(self.param2id.keys())
        return s + str(vals)


class DiscreteHyperParameter(AbstractHyperParameter):
    """
    Discrete Hyper Parameter that defines a set of discrete values that it can take.

    # Arguments:
        name (str): Name of the hyper parameter.
        values (list): A list of values (must all be pickle-able and hashable)
            values or None.

    # Raises:
        ValueError: If the values provided is `None` or length of values is 0.

    # Raises:
        ValueError: If the `name` is not specified.
    """
    def __init__(self, name, values):

        super(DiscreteHyperParameter, self).__init__(name, values)

        if values is not None and len(values) != 0:
            super(DiscreteHyperParameter, self)._build_maps(values)
        else:
            raise ValueError("DiscreteHyperParamter must be passed at least one "
                             "or more values")

    def sample(self):
        """
        Samples a single value from its set of discrete values.

        # Returns:
            a single value from its list of possible values.
        """
        choice = np.random.randint(0, self.num_choices, size=1, dtype=np.int64)[0]
        param = self.id2param[choice]
        return param

    def encode(self, x):
        """
        Encodes a single value into an integer index.

        # Arguments:
            x (int | float | str): A value sampled from its possible values.

        # Returns:
             int value representing its encoded index.
        """
        x = self._cast(x)
        return self.param2id[x]

    def decode(self, x):
        """
        Decodes a single encoded integer into its original value.

        # Args:
            x (int): an integer encoded value.

        # Returns:
             (int | float | str) representing the actual decoded value.
        """
        param = self.id2param[x]
        return self._cast(param)

    def _cast(self, x):
        """
        Casts the sample to its original data type.

        # Arguments:
            x (int | float | str): Input sample that will be cast to the
                correct data type.

        # Returns:
            the sample cast to the correct data type.
        """
        return self.param2type[x](x)

    def get_config(self):
        """
        Creates the config of the class with all of its values.

        # Returns:
            a dictionary with the config of the class.
        """
        config = {
            'values': list(self.id2param.values()),
        }

        base_config = super(DiscreteHyperParameter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AbstractContinuousHyperParameter(AbstractHyperParameter):
    """
    An abstract hyper parameter that represents a parameter that can take a range
    of values from a certain distribution.

    # Arguments:
        name (str): Name of the parameter.
        val1 (float): A symbolic value that is used by subclasses.
        val2 (float): A symbolic value that is used by subclasses.
        log_encode (bool): Determines whether the encoding must be in natural
            log-space or not.

    # Raises:
        NotImplementedError: If `sample()` is called.
    """
    def __init__(self, name, val1, val2, log_encode=False):
        super(AbstractContinuousHyperParameter, self).__init__(name, None)

        if val1 is not None and val2 is not None:
            self._val1 = float(val1)
            self._val2 = float(val2)
        else:
            raise ValueError("val1 and val2 must be floating point "
                             "numbers for ContinuousHyperParameters")

        self.log_encode = log_encode

        if log_encode:
            if val1 < 0.0:
                raise ValueError("When using log encoding, negative values are not allowed for parameters")

    def sample(self):
        """
        Abstract method that must be redefined by base classes.

        # Returns:
            a float value.
        """
        raise NotImplementedError("Subclass must implement this method !")

    def encode(self, x):
        """
        Encodes the floating point value into log space if `log_space` was set in
        the constructor, else returns its original value.

        # Arguments:
            x (float): a single sample.

        # Returns:
             float.
        """
        x = self._cast(x)

        if self.log_encode:
            x = self._cast(np.log(x))

        return x

    def decode(self, x):
        """
        Decodes the floating point value into normal space if `log_space` was set in
        the constructor, else returns its original value.

        # Arguments:
            x (float): a single encoded sample.

        # Returns:
             float.
        """
        x = self._cast(x)

        if self.log_encode:
            x = self._cast(np.exp(x))

        return x

    def _cast(self, x):
        """
        Casts the sample to its original data type.

        # Arguments:
            x (int | float | str): Input sample that will be cast to the
                correct data type.

        # Returns:
            the sample cast to the correct data type.
        """
        return float(x)

    def get_config(self):
        """
        Creates the config of the class with all of its values.

        # Returns:
            a dictionary with the config of the class.
        """
        base_config = super(AbstractContinuousHyperParameter, self).get_config()
        return base_config

    def __repr__(self):
        s = "%s : continuous [%0.3f, %0.3f)\n" % (self.name, self._val1, self._val2)
        return s


class UniformContinuousHyperParameter(AbstractContinuousHyperParameter):
    """
    A hyper parameter that represents a parameter that can take a range
    of values from a uniform distribution.

    # Arguments:
        name (str): Name of the parameter.
        min_value (float): The minimum value (inclusive) that the uniform
            distribution can take.
        max_value (float): The maximum value (exclusive) that the uniform
            distribution can take.
        log_encode (bool): Determines whether the encoding must be in natural
            log-space or not.
    """
    def __init__(self, name, min_value, max_value, log_encode=False):

        super(UniformContinuousHyperParameter, self).__init__(name, min_value, max_value, log_encode)

    def sample(self):
        """
        Samples uniformly from the range [min_value, max_value).

        # Returns:
            float.
        """
        value = np.random.uniform(self._val1, self._val2, size=1)[0]
        return value

    def get_config(self):
        """
        Creates the config of the class with all of its values.

        # Returns:
            a dictionary with the config of the class.
        """
        config = {
            'min_value': self.min_value,
            'max_value': self.max_value,
            'log_encode': self.log_encode,
        }

        base_config = super(UniformContinuousHyperParameter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def min_value(self):
        return self._val1

    @property
    def max_value(self):
        return self._val2


class NormalContinuousHyperParameter(AbstractContinuousHyperParameter):
    """
    A hyper parameter that represents a parameter that can take a range
    of values from a normal distribution.

    # Arguments:
        name (str): Name of the parameter.
        mean (float): The mean of the normal distribution.
        std (float): The standard deviation of the normal distribution.
    """
    def __init__(self, name, mean, std):
        super(NormalContinuousHyperParameter, self).__init__(name, mean, std, False)

    def sample(self):
        """
        Samples from the normal distribution with a mean and standard deviation
        as specified in the constructor.

        # Returns:
            float.
        """
        value = np.random.normal(self._val1, self._val2, size=1)[0]
        return value

    def get_config(self):
        """
        Creates the config of the class with all of its values.

        # Returns:
            a dictionary with the config of the class.
        """
        config = {
            'mean': self.mean,
            'std': self.std,
        }

        base_config = super(NormalContinuousHyperParameter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def mean(self):
        return self._val1

    @property
    def std(self):
        return self._val2


class HyperParameterList(AbstractHyperParameter):
    """
    A composite hyper parameter, that encloses a list of hyper parameters
    (either discrete or continuous) and provides utility methods for efficient
    handling by the engine.

    # Arguments:
        hyper_parameter_list (list(AnstractHyperParameter) | None): A list of
            hyper parameters or None (which initializes this with 0 elements).
    """
    def __init__(self, hyper_parameter_list=None):
        super(HyperParameterList, self).__init__('parameter_list', None)
        self.name_map = OrderedDict()

        self._build_maps(hyper_parameter_list)

    def sample(self):
        """
        Samples all of its component parameters and returns a list of the samples.

        # Returns:
             list of sampled parameters.
        """
        values = []
        for v in self.id2param.values():  # type: AbstractHyperParameter
            x = v.sample()
            values.append(x)
        return values

    def encode(self, x):
        """
        Encodes a list of sampled hyper parameters.

        # Arguments:
            x (list | np.ndarray): A python list or numpy array of samples
                from the list of hyper parameters.

        # Raises:
            ValueError: If a numpy array of more than 1 dimension is provided.

        # Returns:
             ndarray(float).
        """
        if isinstance(x, np.ndarray):
            if x.ndim != 1:
                raise ValueError("When encoding a list of hyper parameters, provide a python list "
                                 "or 1-dim numpy array")
            else:
                x = x.tolist()

        values = []
        for data, param in zip(x, self.id2param.values()):  # type: (AbstractHyperParameter)
            v = param.encode(data)
            values.append(v)

        values = np.array(values)
        return values

    def decode(self, x):
        """
        Decodes a list of sampled hyper parameters.

        # Arguments:
            x (list(int | float)): a list of encoded integer or floating point
                values that are to be decoded.

        # Returns:
             list of decoded samples.
        """
        if isinstance(x, np.ndarray):
            if x.ndim != 1:
                raise ValueError("When encoding a list of hyper parameters, provide a python list "
                                 "or 1-dim numpy array")
            else:
                x = x.tolist()

        values = []
        for data, param in zip(x, self.id2param.values()):  # type: (AbstractHyperParameter)
            v = param.decode(data)
            v = param._cast(v)
            values.append(v)

        return values

    def _build_maps(self, values):
        """
        Adds the individual hyper parameters to the list.

        # Arguments:
            values (list(AbstractHyperParameter) | None): a list of parameters.
        """
        if values is not None:
            for param in values:  # type: AbstractHyperParameter
                self.add_hyper_parameter(param)

    def _cast(self, x):
        """
        Casts all of the samples to their original data types.

        # Arguments:
            x (list): Input samples that will be cast to their
                correct data types.

        # Returns:
            the list of samples cast to their correct data types.
        """
        if isinstance(x, np.ndarray):
            if x.ndim != 1:
                raise ValueError("When encoding a list of hyper parameters, provide a python list "
                                 "or 1-dim numpy array")
            else:
                x = x.tolist()

        types = []
        for data, param in zip(x, self.id2param.values()):  # type: (AbstractHyperParameter)
            v = param._cast(data)
            types.append(v)

        return types

    def get_config(self):
        """
        Creates the config of the class with all of its values.

        # Returns:
            an ordered dictionary with the config of the class.
        """
        config = OrderedDict()

        for name, param in zip(self.name_map.values(), self.id2param.values()):  # type: (AbstractHyperParameter)
            class_name = param.__class__.__name__
            param_config = param.get_config()
            config[name] = [class_name, param_config]

        return config

    @classmethod
    def load_from_config(cls, config):
        params = []

        for name, cls_config in config.items():
            param_class_name, param_config = cls_config
            param_class = get_parameter(param_class_name)
            param = param_class(**param_config)

            params.append(param)

        return cls(params)

    def add_hyper_parameter(self, parameter):
        """
        Adds a single hyper parameter (discrete or continuous) to the list
        of hyper parameters managed by this HyperParameterList.

        # Arguments:
            parameter (AbstractHyperParameter): a subclass of AbstractHyperParameter,
                which will be embedded into this composite class.

        # Raises:
            ValueError: If the passed parameter is `None`, or the name already
                exists in the list of managed parameters.
        """
        if parameter is None:
            raise ValueError("When adding a hyper parameter, `None` cannot be passed")

        if parameter.name in self.name_map.values():
            raise ValueError('Cannot add two hyper parameters with same name (%s)' %
                             parameter.name)

        id = str(uuid.uuid4())

        self.name_map[id] = parameter.name
        self.id2param[id] = parameter
        self.param2id[parameter.name] = id
        self.num_choices += 1

    def remove_hyper_parameter(self, parameter):
        """
        Removes a single hyper parameter (discrete or continuous) from the list
        of hyper parameters managed by this HyperParameterList.

        # Arguments:
            parameter (AbstractHyperParameter, str): A string name or a subclass
                of AbstractHyperParameter which needs to be removed.

        # Raises:
            ValueError: If the passed parameter is `None`.
        """
        if parameter is None:
            raise ValueError("When adding a hyper parameter, `None` cannot be passed")

        if isinstance(parameter, AbstractHyperParameter):
            id = self.param2id[parameter.name]
            del self.param2id[parameter.name]
        else:
            if parameter in self.param2id:
                id = self.param2id[parameter]
                del self.param2id[parameter]
            else:
                raise KeyError("The hyper parameter with name %s has not been added to "
                               "this list." % parameter)

        del self.name_map[id]
        del self.id2param[id]
        self.num_choices -= 1

    def get_parameter_names(self):
        """
        Gets a list of all the parameter names managed by this class.

        # Returns:
            a list(str) with the names of the parameters.
        """
        name_list = []
        for v in self.id2param.values():  # type: AbstractHyperParameter
            name_list.append(v.name)
        return name_list

    def __repr__(self):
        s = ""
        for v in self.id2param.values():  # type: AbstractHyperParameter
            s = s + str(v) + "\n"
        return s

    def __len__(self):
        return len(self.name_map)


def set_custom_parameter_class(cls):
    """
    Utility function to dynamically add a custom hyper parameter
    to the set of available hyper parameters.

    # Arguments:
        cls (cls): A class which extends `AbstractHyperParameter` in some way
            and implements the abstract methods.
    """
    global _CUSTOM_PARAMETERS
    _CUSTOM_PARAMETERS[cls.__name__] = cls


def get_parameter(name):
    """
    Utility method to get the hyper parameter class by its name.

    # Arguments:
        name (str): Name of the class or its alias.

    # Raises:
        ValueError: If the class with the provided name does not exists in
            the set of available parameters.

    # Returns:
        The hyper parameter class.
    """
    global _CUSTOM_PARAMETERS

    if name in _CUSTOM_PARAMETERS:
        return _CUSTOM_PARAMETERS[name]

    module_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    module_classes = dict(module_classes)

    if name in module_classes:
        return module_classes[name]
    else:
        raise ValueError('No hyper parameter class with the name %s was found in '
                         'the hyper parameters module !')


# Aliases
DiscreteHP = DiscreteHyperParameter
UniformHP = UniformContinuousHyperParameter
NormalHP = NormalContinuousHyperParameter
