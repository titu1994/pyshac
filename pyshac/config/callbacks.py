""" Callback module is adapted from the Keras library """
import six
import csv
import io
import os

import numpy as np
from collections import Iterable
from collections import OrderedDict

# compatible with both Python 2 and 3
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def get_history(callbacks):
    """
    Gets the History callback from a list of callbacks.

    # Argumetns:
        callbacks (list | CallbackList): a list of callbacks

    # Returns:
        A History callback object or None if it was not found.
    """
    history = None

    for c in callbacks:
        if isinstance(c, History):
            history = c
            break

    return history


class Callback(object):
    """
    Abstract base class used to build new callbacks.

    # Properties
        engine: instance of a PySHAC Engine.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.
    """
    def __init__(self):
        self.engine = None

    def set_engine(self, engine):
        """
        Sets an instance of a PySHAC Engine.

        # Arguments:
            engine (AbstractSHAC): A concrete implementation of the
                SHAC engine.
        """
        self.engine = engine

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        # Arguments
            logs (dict | None): dictionary of logs.
        """
        pass

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        # Arguments
            logs (dict | None): dictionary of logs.
        """
        pass

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the start of an epoch.

        # Arguments
            epoch (int): index of epoch.
            logs (dict | None): dictionary of logs.
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch.

        # Arguments
            epoch (int): index of epoch.
            logs (dict | None): dictionary of logs.
        """
        pass

    def on_evaluation_begin(self, params, logs=None):
        """
        Called before the generated parameters are evaluated.

        # Arguments:
            params (list(OrderedDict)): A list of OrderedDicts,
                such that each item is a dictionary of the names
                and sampled values of a HyperParemeterList.
            logs (dict | None): dictionary of logs.
        """
        pass

    def on_evaluation_ended(self, evaluations, logs=None):
        """
        Called after the generated parameters are evaluated.

        # Arguments:
            evaluations (list(float)): A list of floating point
                values, corresponding to the provided parameter
                settings.
            logs (dict | None): dictionary of logs.
        """
        pass

    def on_dataset_changed(self, dataset, logs=None):
        """
        Called with the dataset maintained by the engine is
        updated with new samples or data.

        # Arguments:
            dataset (Dataset): A Dataset object which contains
                the history of sampled parameters and their
                corresponding evaluation values.
            logs (dict | None): dictionary of logs.
        """
        pass


class CallbackList(Callback):
    """
    Container abstracting a list of callbacks.

    Automatically creates a History callback if not provided in
    list of callbacks.

    # Arguments
        callbacks (list | None): List of `Callback` instances.
    """
    def __init__(self, callbacks=None):
        super(CallbackList, self).__init__()

        callbacks = callbacks or []

        # check if list has History callback in it
        history = get_history(callbacks)
        if history is None:
            callbacks.append(History())

        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        """
        Appends an additional callback to the callback list.

        # Arguments:
            callback (Callback):
        """
        self.callbacks.append(callback)

    def set_engine(self, engine):
        """
        Sets an instance of a PySHAC Engine.

        # Arguments:
            engine (AbstractSHAC): A concrete implementation of the
                SHAC engine.
        """
        for callback in self.callbacks:
            callback.set_engine(engine)

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        # Arguments
            logs (dict | None): dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        # Arguments
            logs (dict | None): dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the start of an epoch.

        # Arguments
            epoch (int): index of epoch.
            logs (dict | None): dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch.

        # Arguments
            epoch (int): index of epoch.
            logs (dict | None): dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_evaluation_begin(self, params, logs=None):
        """
        Called before the generated parameters are evaluated.

        # Arguments:
            params (list(OrderedDict)): A list of OrderedDicts,
                such that each item is a dictionary of the names
                and sampled values of a HyperParemeterList.
            logs (dict | None): dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_evaluation_begin(params, logs)

    def on_evaluation_ended(self, evaluations, logs=None):
        """
        Called after the generated parameters are evaluated.

        # Arguments:
            evaluations (list(float)): A list of floating point
                values, corresponding to the provided parameter
                settings.
            logs (dict | None): dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_evaluation_ended(evaluations, logs)

    def on_dataset_changed(self, dataset, logs=None):
        """
        Called with the dataset maintained by the engine is
        updated with new samples or data.

        # Arguments:
            dataset (Dataset): A Dataset object which contains
                the history of sampled parameters and their
                corresponding evaluation values.
            logs (dict | None): dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_dataset_changed(dataset, logs)

    def __iter__(self):
        return iter(self.callbacks)


class History(Callback):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every engine. The `History` object
    gets returned by the `fit` or `fit_dataset` methods.
    """
    def __init__(self):
        super(History, self).__init__()

    def on_train_begin(self, logs=None):
        """
        Initializes the epoch list and history dictionary.

        # Arguments:
            logs (dict | None): dictionary of logs.
        """
        self.epochs = []
        self.history = logs or {}

    def on_epoch_end(self, epoch, logs=None):
        """
        Adds the current epoch's log values to the history.

        # Arguments:
            epoch (int): index of epoch.
            logs (dict | None): dictionary of logs.
        """
        logs = logs or {}
        self.epochs.append(epoch)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class CSVLogger(Callback):
    """
    Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    csv_logger = CSVLogger('training.log')
    shac.fit(evaluation_function, callbacks=[csv_logger])
    ```

    # Arguments
        filename (str): filename of the csv file, e.g. 'run/log.csv'.
        separator (str): string used to separate elements in the csv file.
        append (bool): True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """
    def __init__(self, filename, separator=',', append=False):
        super(CSVLogger, self).__init__()
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)

            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
