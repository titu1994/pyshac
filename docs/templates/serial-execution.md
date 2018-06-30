# Serial Execution of Evaluation Functions
----

For all engines, to be efficient and reduce execution time, sample generation, and training of models is done using
Joblib and Loky. However, the evaluation function cannot be managed by the training module, since it is a function written by the user.

As such, to offer maximum flexibility, we offer two alternatives :

- Serial Evaluation
- Managed Evaluations

## Serial Evaluation

Due to the need for forcing serial execution of the evaluation functions, there are 2 parameters exposed for all
engines :

- `num_parallel_evaluators` : Set this to `1` for serial execution of the evaluation function
- `evaluator_backend` : This is generally set to `loky`, but should be set to `threading` for serial execution.

```python

shac = SHAC(eval_fn, [params...], total_budget, batch_size)

# This sets the engine to serial execution
shac.set_num_parallel_evaluators(1)
shac.concurrent_evaluators()

```

Similarly, if for some reason you wish to force serial execution of the generators, there are 2 parameters exposed fpr all
engines :

- `num_parallel_generators` : Set this to `1` for serial execution of the evaluation function
- `generator_backend` : This is generally set to `loky`, but should be set to `threading` for serial execution.

```python

shac = SHAC(eval_fn, [params...], total_budget, batch_size)

# This sets the engine to serial execution
shac.set_num_parallel_generators(1)
shac.parallel_evaluators()

```

!!!info "Helper functions for backends"
    All engines support two methods : `parallel_evaluators()` and `concurrent_evaluators()`.
    Using this will set the engine to use the `loky` and `threading` backend respectively.

    Due to Python's Global Interpreter Lock, it is possible to use the `threading` backend to
    reduce the number of parallel executions. However, these models are still executed concurrently,
    therefore it must be ensured that the evaluators do not exhaust the system RAM / GPU RAM.

## Managed Evaluations

When using PySHAC for neural network hyper parameter searches or architecture generation, it would be extremely slow to
evaluate each model sequentially.

Therefore, there are extensions to the core engine, called `TensorflowSHAC` and `KerasSHAC`. These provide a managed backend,
create seperate graphs for each core of execution and offer some parallel evaluation management.

- [TorchSHAC](core/torch_engine.md#torchshac) : Provides a wrapper over `SHAC` so as to provide a unified interface alongside
the other managed session. Makes it simpler to explicitly set the parallelism of the evaluators.

- [TensorflowSHAC](core/tf_engine.md#tensorflowshac) : Provides a `tf.Session` alongside the worker id and the sampled parameters. This session is wrapped
by the evaluation, so all tensors created inside the evaluation function can be run using this session. The graph can be
obtained using the generic `tf.get_default_graph()`, if required.

- [KerasSHAC](core/keras_engine.md#kerasshac) : Provides a managed session similar to `TensorflowSHAC`, but can be possibly used with other backends. Since
Theano does not support memory management (deletion of old models), only Tensorflow and CNTK backends are supported for
the moment.
