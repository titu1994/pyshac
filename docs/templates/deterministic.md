# Deterministic Engines
----

`PySHAC`, on its own, uses no random seed to enable deterministic training or evaluation.
However, there are often the cases where experiments must be controlled, and it makes sense
to set the random seed for deterministic behaviour.

!!!warning
    PySHAC ignores the global numpy random seed. Therefore setting `np.random.seed(...)`
    or `np.random.RandomState(...)` will **not** make the PySHAC engine or its components
    deterministic.

## Deterministic Behaviour for the Engine
----

Deterministic behaviour can be set in the engine either globally (all operations on the engine are
deterministic) or locally (engine is deterministic only inside the context block).

### Setting Global Deterministic Behaviour

Given we have initialized an engine as below (we use the basic `SHAC` engine here, but any engine
subclass can use the same API), we enable global deterministic behaviour as follows :

```python
import pyshac

shac = pyshac.SHAC(...)

# make engine deterministic
shac.set_seed(seed)  # an integer seed
```

Once the seed has been set as such, all operations of the engine are deterministic globally until
either the engine is destroyed or we re-enable non-determinism in the engine as follows :

```python
shac.set_seed(None)  # use `None` to make the engine non deterministic again.
```

### Setting Local Deterministic Behaviour
----

Given we have initialized an engine as below (we use the basic `SHAC` engine here, but any engine
subclass can use the same API), we enable local deterministic behaviour using a context manager as
follows :

```python
import pyshac

shac = pyshac.SHAC(...)

with shac.as_deterministic(seed):  # deterministic only within this block
    ...
```

## Restoring Engines with Deterministic Behaviour
----

Engines can be saved and then restored as required by the engine, but by default, even restored engines
are non-deterministic. Deterministic behaviour must be reestablished, but in the proper order.

```python

# First create a default engine without any hyper parameters.
new_shac = pyshac.SHAC(None, total_budget, batch_size=batch_size, objective=objective)

# Restore the engine
new_shac.restore_data()

# Restore deterministic behaviour now.
new_shac.set_seed(seed)
```

**NOTE**:
----
If the user attempts to set the seed before the engine has been restored, then the engine will issue
a warning stating that the engine had no hyper parameters which it could seed.

# Deterministic Managed Engines
----

The managed engines which subclass `SHAC` also have full support for deterministic behaviour.

This entails :

- `TensorflowSHAC`: Seeds the engine randomness as well as uses `tf.set_random_seed(seed)` to make the engine deterministic.
- `KerasSHAC`: Seeds the engine randomness as well as the Tensorflow seed when using the Tensorflow backend.
- `TorchSHAC`: Seeds the engine randomness as well as uses `np.random.seed()`, `torch.manual_seed()` and `torch.cuda.manual_seed_all()`.
