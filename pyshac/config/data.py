import codecs
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyshac.config.hyperparameters as hp

# compatible with both Python 2 and 3
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class Dataset(object):
    """Dataset manager for the engines.

        Holds the samples and their associated evaluated values in a format
        that can be serialized / restored as well as encoder / decoded for
        training.

        # Arguments:
            parameter_list (hp.HyperParameterList | list | None): A python list
                of Hyper Parameters, or a HyperParameterList that has been built.
                Can also be None, if the parameters are to be assigned later.
    """
    def __init__(self, parameter_list=None):

        if not isinstance(parameter_list, hp.HyperParameterList):
            if type(parameter_list) == list or type(parameter_list) == tuple:
                parameter_list = hp.HyperParameterList(parameter_list)

        self._parameters = parameter_list
        self.X = []
        self.Y = []
        self.size = 0

        self.basedir = 'shac'
        self._prepare_dir()

    def add_sample(self, parameters, value):
        """
        Adds a single row of data to the dataset.
        Each row contains the hyper parameter configuration as well as its associated
        evaluation measure.

        # Arguments:
            parameters (list): A list of hyper parameters that have been sampled
            value (float): The evaluation measure for the above sample.

        """
        self.X.append(parameters)
        self.Y.append(value)
        self.size += 1

    def clear(self):
        """
        Removes all the data of the dataset.
        """
        self.X = []
        self.Y = []
        self.size = 0

    def encode_dataset(self, X=None, Y=None, objective='max'):
        """
        Encode the entire dataset such that discrete hyper parameters are mapped
        to integer indices and continuous valued hyper paramters are left alone.

        # Arguments
            X (list | np.ndarray | None): The input list of samples. Can be None,
                in which case it defaults to the internal samples.
            Y (list | np.ndarray | None): The input list of evaluation measures.
                Can be None, in which case it defaults to the internal evaluation
                values.
            objective (str): Whether to maximize or minimize the
                value of the labels.

        # Raises:
            ValueError: If `objective` is not in [`max`, `min`]

        # Returns:
             A tuple of numpy arrays (np.ndarray, np.ndarray)
        """
        if objective not in ['max', 'min']:
            raise ValueError("Objective must be in `max` or `min`")

        if X is None:
            X = self.X

        if Y is None:
            Y = self.Y

        encoded_X = []
        for x in X:
            ex = self._parameters.encode(x)
            encoded_X.append(ex)

        encoded_X = np.array(encoded_X)

        y = np.array(Y)
        median = np.median(y)

        encoded_Y = np.sign(y - median)

        if objective == 'max':
            encoded_Y = np.where(encoded_Y <= 0., 0.0, 1.0)
        else:
            encoded_Y = np.where(encoded_Y >= 0., 0.0, 1.0)

        return encoded_X, encoded_Y

    def decode_dataset(self, X=None):
        """
        Decode the input samples such that discrete hyper parameters are mapped
        to their original values and continuous valued hyper paramters are left alone.

        # Arguments:
            X (np.ndarray | None): The input list of encoded samples. Can be None,
                in which case it defaults to the internal samples, which are encoded
                and then decoded.

        # Returns:
             np.ndarray
        """
        if X is None:
            X, _ = self.encode_dataset(self.X)

        decoded_X = []
        for x in X:
            dx = self._parameters.decode(x)
            decoded_X.append(dx)

        decoded_X = np.array(decoded_X, dtype=np.object)

        return decoded_X

    def save_dataset(self):
        """
        Serializes the entire dataset into a CSV file saved at the path
        provide by `data_path`. Also saves the parameters (list of hyper parameters).

        # Raises:
            ValueError: If trying to save a dataset when its parameters have not been
                set.
        """
        if self._parameters is None:
            raise ValueError("Cannot save a dataset whose parameters have not been set !")

        print("Serializing dataset...")

        x, y = self.get_dataset()
        name_list = self._parameters.get_parameter_names() + ['scores']
        y = y.reshape((-1, 1))

        dataset = np.concatenate((x, y), axis=-1)

        # serialize the data
        df = pd.DataFrame(dataset, columns=name_list)
        df.to_csv(self.data_path, encoding='utf-8', index=True, index_label='id')

        # serialize the parameters
        param_config = self._parameters.get_config()

        with codecs.open(self.parameter_path, 'w', encoding='utf-8') as f:
            json.dump(param_config, f, indent=4)

        print("Serialization of dataset done !")

    def restore_dataset(self):
        """
        Restores the entire dataset from a CSV file saved at the path provided by
        `data_path`. Also loads the parameters (list of hyperparameters).

        # Raises:
            FileNotFoundError: If the dataset is not at the provided path.
        """
        print("Deserializing dataset...")

        df = pd.read_csv(self.data_path, header=0, encoding='utf-8')

        cols = df.columns.values.tolist()
        df.drop(cols[0], axis=1, inplace=True)

        x = df[cols[1:-1]].values
        y = df[cols[-1]].values

        self.set_dataset(x.tolist(), y.tolist())

        with codecs.open(self.parameter_path, 'r', encoding='utf-8') as f:
            param_config = json.load(f, object_pairs_hook=OrderedDict)
            self._parameters = hp.HyperParameterList.load_from_config(param_config)

        print("Deserialization of dataset done ! ")

    def get_dataset(self):
        """
        Gets the entire dataset as a numpy array.

        # Returns:
            (np.ndarray, np.ndarray)
        """
        x = np.array(self.X, dtype=np.object)
        y = np.array(self.Y, dtype=np.object)
        return x, y

    def set_dataset(self, X, Y):
        """
        Sets a numpy array as the dataset.

        # Arguments:
            X (list | tuple | np.ndarray): A numpy array or python list/tuple that contains
                the samples of the dataset.
            Y (list | tuple | np.ndarray): A numpy array or python list/tuple that contains
                the evaluations of the dataset.
        """
        if type(X) != list:
            if isinstance(X, np.ndarray):
                X = X.tolist()
            elif type(X) == tuple:
                X = list(X)
            else:
                raise TypeError("X must be a python list or numpy array")

        if type(Y) != list:
            if isinstance(Y, np.ndarray):
                Y = Y.tolist()
            elif type(Y) == tuple:
                Y = list(Y)
            else:
                raise TypeError("Y must be a python list or numpy array")

        self.X = X
        self.Y = Y
        self.size = len(Y)

    def prepare_parameter(self, sample):
        """
        Wraps a hyper parameter sample list with the name of the
        parameter in an OrderedDict.

        # Arguments:
            sample (list): A list of sampled hyper parameters

        # Returns:
             OrderedDict(str, int | float | str)
        """
        param = OrderedDict()

        for name, value in zip(self._parameters.name_map.values(), sample):
            param[name] = value

        return param

    @property
    def parameters(self):
        """
        Returns the hyper parameter list manager

        # Returns:
            HyperParameterList
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        """
        Sets the hyper parameter list manager

        # Arguments:
            parameters (hp.HyperParameterList | list): a Hyper Parameter List
                or a python list of Hyper Parameters.
        """
        if not isinstance(parameters, hp.HyperParameterList):
            parameters = hp.HyperParameterList(parameters)

        self._parameters = parameters

    def get_parameters(self):
        """
        Gets the hyper parameter list manager

        # Returns:
            HyperParameterList
        """
        return self._parameters

    def set_parameters(self, parameters):
        """
        Sets the hyper parameter list manager

        # Arguments:
            parameters (hp.HyperParameterList | list): a Hyper Parameter List
                or a python list of Hyper Parameters.
        """
        if not isinstance(parameters, hp.HyperParameterList):
            parameters = hp.HyperParameterList(parameters)

        self._parameters = parameters

    def get_best_parameters(self, objective='max'):
        """
        Selects the best hyper parameters according to the maximization
        or minimization of the objective value.

        Returns `None` if there are no samples in the dataset.

        # Arguments:
            objective: String label indicating whether to maximize or minimize
                the objective value.

        # Raises:
            ValueError: If the objective is not `max` or `min`.

        # Returns:
            A list of hyperparameter settings or `None` if the dataset is empty.
        """
        if objective not in ['max', 'min']:
            raise ValueError("Objective must be one of 'max' or 'min'")

        if self.size == 0:
            return None

        if objective == 'max':
            return self.prepare_parameter(self.X[int(np.argmax(self.Y))])
        else:
            return self.prepare_parameter(self.X[int(np.argmin(self.Y))])

    def _prepare_dir(self):
        """
        Creates the directories needed to save the data and parameters.
        """
        path = os.path.join(self.basedir, 'datasets')
        if not os.path.exists(path):
            os.makedirs(path)

        self.data_path = os.path.join(path, 'dataset.csv')
        self.parameter_path = os.path.join(path, 'parameters.json')
    
    @classmethod
    def load_from_directory(cls, dir='shac'):
        """
        Static method to load the dataset from a directory.

        # Arguments:
            dir: The base directory where 'shac' directory is. It will build the path
                to the data and parameters itself.

        # Raises:
            FileNotFoundError: If the directory does not contain the data and parameters.
        """
        if dir == 'shac':
            dir = os.path.join(dir, 'datasets')

        data_path = os.path.join(dir, 'dataset.csv')
        parameter_path = os.path.join(dir, 'parameters.json')

        if not os.path.exists(data_path) or not os.path.exists(parameter_path):
            raise FileNotFoundError("Files 'dataset.csv' and 'parameters.json' not found at %s" %
                                    dir)

        df = pd.read_csv(data_path, header=0, encoding='utf-8')

        cols = df.columns.values.tolist()
        df.drop(cols[0], axis=1, inplace=True)

        x = df[cols[1:-1]].values
        y = df[cols[-1]].values

        X = x.tolist()
        Y = y.tolist()

        with codecs.open(parameter_path, 'r', encoding='utf-8') as f:
            param_config = json.load(f, object_pairs_hook=OrderedDict)
            parameters = hp.HyperParameterList.load_from_config(param_config)

        obj = cls(parameters)
        obj.set_dataset(X, Y)

        return obj

    def __len__(self):
        return self.size
