# Managed Engines for Advanced Workflows
----

When using just CPU resources, `SHAC` is sufficient for most tasks.

However, when more complicated workflows are involved - such as training models and evaluation with
Tensorflow, PyTorch or Keras or other such libraries that involve graph based execution and memory
management, it is advised to use the Managed Engines detailed here.

There are three engines provided :

- [TorchSHAC](core/torch_engine.md#torchshac)
- [TensorflowSHAC](core/tf_engine.md#tensorflowshac)
- [KerasSHAC](core/keras_engine.md#kerasshac)

## Motivation for Managed Engines
----

- Tensorflow / Keras use graphs to generate and train models. These graphs and their scopes need to be managed properly.
- If evaluators are models which run on the GPU, it may not be possible to run more than 1 model at once due to memory constraints.
- It is necessary to provide a simple interface to decide whether GPUs will be used or not.

!!!note
    It is upto the user to utilize the `worker_id` parameter to determine the placement of the operations.

    The session provided **does not** wrap the evaluation function in a device, as certain ops may not be
    available on GPUs.

    In the same manner, when using PyTorch, the device is not set by the managed engine. It is the task of
    the user to utilize the provided `worker_id` in pushing models onto the correct devices.

## [TorchSHAC](core/torch_engine.md#torchshac)
----

This is the most basic managed engine, which only provides a unified interface similar to `TensorflowSHAC` and
`KerasSHAC`. In term's of functionality, it does not manage memory or graph execution, since PyTorch does not
utilize a graph in a user facing way.

It simply provides a way to determine whether / how many evaluation processs can be started using the `max_gpu_evaluators`
argument.

As such, when using dynamic execution environments, such as `PyTorch`, `Tensorflow Eager` or `tf.keras` with
Eager Execution enabled, it is suggested to use this backend.

Like `SHAC`, the evaluation function will receive 2 inputs :

- **Worker ID**: Integer id that can be left alone when executing only on CPU or used to determine the GPU id on which the model will be evaluated.
- **Parameter OrderedDict**: An OrderedDict which contains the (name, value) pairs of the Parameters passed to the engine.
    -   These are the values of the sampled hyper parameters which have passed through the current cascade of models.


## [TensorflowSHAC](core/tf_engine.md#tensorflowshac)

Due to its graph based execution, Tensorflow is the primary candidate to use a managed session.

Unlike the `SHAC` and `TorchSHAC` engines, an evaluation function using the `TensorflowSHAC` engine will receive
3 inputs :

- **Tensorflow Session**: This session manages a separate graph for each parallel evaluator.
    -   The scope of all ops inside the evaluation function is the scope of the managed graph.
    -   It can be used to run all operations inside of the evaluation function.
- **Worker Id**: An integer id that can be used alongside multiple GPU's to determine which GPU will be used to
- **Parameter OrderedDict**: An OrderedDict which contains the (name, value) pairs of the Parameters passed to the engine.
    -   These are the values of the sampled hyper parameters which have passed through the current cascade of models.

Since these graphs are managed by the engine, once the evaluation function has provided a value, the respective graph
will be destroyed and resources released automatically by the engine.

## [KerasSHAC](core/keras_engine.md#kerasshac)

Since Keras is a wrapper over Tensorflow and CNTK which support multiple GPU execution, this wrapper simply
provides a thin wrapper over the `TensorflowSHAC` engine. It's primary purpose is to release resources
after the evaluation of the function.

Unlike the `SHAC` and `TorchSHAC` engines, an evaluation function using the `KerasSHAC` engine will receive
3 inputs :

- **Tensorflow Session or None**: This session manages a separate graph for each parallel evaluator.
    -   The scope of all ops inside the evaluation function is the scope of the managed graph.
    -   It can be used to run all operations inside of the evaluation function.
    -   When using the CNTK backend, this argument will be `None`.
- **Worker Id**: An integer id that can be used alongside multiple GPU's to determine which GPU will be used to
- **Parameter OrderedDict**: An OrderedDict which contains the (name, value) pairs of the Parameters passed to the engine.
    -   These are the values of the sampled hyper parameters which have passed through the current cascade of models.

Since these graphs are managed by the engine, once the evaluation function has provided a value, the respective graph
will be destroyed and resources released automatically by the engine.