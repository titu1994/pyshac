# Callbacks
------

Callbacks can be used with PySHAC for custom code execution, such as monitoring the improvement of the engine,
maintaining a history of the training session or simply writing the logs to files.

Callbacks allow you the flexibility to prepare your state prior to training or evaluation, and in doing so can
allow one to perform stateful evaluation if necessary using concurrent evaluators.

## Usage

Callbacks can be imported from the `pyshac.config.callbacks` package as shown below.

```python

from pyshac.config.callbacks import History, CSVLogger

shac = SHAC(...)

# History is not needed here, as it is automatically added by default for all .fit / .fit_dataset calls.
callbacks = [History(), CSVLogger('path/to/file.csv')]

history = shac.fit(evaluation_function, callbacks=callbacks)
OR
history = shac.fit_dataset('path/to/dataset', callbacks=callbacks)

print(history.history)
```

## History

All calls to `shac.fit` and `shac.fit_dataset` will now return a `History` object, which is a callback to
monitor and log all valuable information occurring during training.

The `History` object has a special member, also called `history`, which is a dictionary containing all of
the logged values.

!!!info "History is added by default"
    The `History` callback is added by default to all calls to `fit` or `fit_dataset` and therefore
    there it is not necessary to add this callback manually.