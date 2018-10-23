import numpy as np
import pyshac.core.engine as optimizer


class TensorflowSHAC(optimizer._SHAC):
    """
    SHAC Engine specifically built for the Graph based workflow of Tensorflow.

    It wraps the abstract SHAC engine with utilities to improve workflow with Tensorflow,
    and performs additional maintenance over the evaluation function, such as creating a
    graph and session for it, and then destroying it and releasing its resources once
    evaluation is over.

    This engine is suitable for Tensorflow based work. Since tensorflow allocates
    memory to the graph, graph maintanence is managed as much as possible. As long
    as the system has enough memory to run multiple copies of the evaluation model,
    there is no additional work required by the user inside the evaluation function.

    Note : When using Eager Execution, it is preferred to use the default `SHAC` engine,
    as memory management is done by Tensorflow automatically in such a scenario.

    # Arguments:
        hyperparameter_list (hp.HyperParameterList | None): A list of parameters
            (or a HyperParameterList) that are passed to define the search space.
            Can be None, so that it is loaded from disk instead.
        total_budget (int): `N`. Defines the total number of models to evaluate.
        num_batches (int): `M`. Defines the number of batches the work is distributed
            to. Must be set such that `total budget` is divisible by `batch size`.
        max_gpu_evaluators (int): number of gpus. Can be 0 or more. Decides the number of
            GPUs used to evaluate models in parallel.
        objective (str): Can be `max` or `min`. Whether to maximise the evaluation
            measure or minimize it.
        max_classifiers (int): Maximum number of classifiers that
            are trained. Default (18) is according to the paper.
        max_cpu_evaluators (int): Positive integer > 0 or -1. Sets the number
            of parallel evaluation functions calls are executed simultaneously.
            Set this to 1 unless you have a lot of memory for 2 or more models
            to be trained simultaneously. If set to -1, uses all CPU cores to
            evaluate N models simultaneously. Will cause OOM if the models are
            large.

    # References:
        - [Parallel Architecture and Hyperparameter Search via Successive Halving and Classification](https://arxiv.org/abs/1805.10255)

    # Raises:
        ValueError: If `total budget` is not divisible by `batch size`.
    """
    def __init__(self, hyperparameter_list, total_budget, num_batches,
                 max_gpu_evaluators, objective='max', max_classifiers=18,
                 max_cpu_evaluators=1):

        super(TensorflowSHAC, self).__init__(hyperparameter_list, total_budget,
                                             num_batches=num_batches, objective=objective,
                                             max_classifiers=max_classifiers)

        self.max_gpu_evaluators = max_gpu_evaluators

        if self.max_gpu_evaluators == 0:  # CPU only
            # By default, allow only 1 evaluation at a time
            self.num_parallel_evaluators = max_cpu_evaluators
        else:
            # CPU and GPU. Limit number of parallel GPU calls
            self.num_parallel_evaluators = max_gpu_evaluators
            self.limit_memory = True

    def _evaluation_handler(self, func, worker_id, parameter_dict, *batch_args):
        """
        A wrapper over the evaluation function of the user, so as to manage the
        Tensorflow session and graph, and perform additional cleanup such as
        memory releasing and graph deletion etc.

        # Arguments:
            func ((tf.Session, int, list) -> float ): The evaluation function is
                passed a managed Tensorflow Session, the integer id (of the worker) and the
                sampled hyper parameters in an OrderedDict. The evaluation function is expected
                to pass a python floating point number representing the evaluated value.
            worker_id (int):  The integer id of the process, ranging from
                [0, num_parallel_evaluator).
            parameter_dict (OrderedDict(str, int | float | str)): An OrderedDict of
                (name, value) pairs where the name and order is based on the declaration
                in the list of parameters passed to the constructor.
            batch_args (list | None): Optional arguments that a subclass can pass to all
                batches if necessary.

        # Returns:
             float representing the evaluated value.
        """
        import tensorflow as tf

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        graph = tf.Graph()
        session = tf.Session(graph=graph, config=config)

        # if tf.keras will be used, this is required, so
        # set it by default.
        tf.keras.backend.set_session(session)

        with graph.as_default():
            if self.max_gpu_evaluators == 0:
                with tf.device('/cpu:0'):
                    output = func(session, worker_id, parameter_dict)
                    output = float(output)
            else:
                output = func(session, worker_id, parameter_dict)
                output = float(output)

        session.close()

        return output

    def fit(self, eval_fn, skip_cv_checks=False, early_stop=False, relax_checks=False,
            callbacks=None):
        """
        Generated batches of samples, trains `total_classifiers` number of XGBoost models
        and evaluates each batch with the supplied function in parallel.

        Allows manually changing the number of processes that are used to generate samples
        or to evaluate them. While the defaults generally work well, further performance
        gains can be had by trying different values according to the limits of the system.

        ```python
        >>> eval = lambda id, params: np.exp(params['x'])
        >>> shac = TensorflowSHAC(params, total_budget=100, num_batches=10)

        >>> shac.set_num_parallel_generators(20)  # change the number of generator process
        >>> shac.set_num_parallel_evaluators(1)  # change the number of evaluator processes
        >>> shac.generator_backend = 'multiprocessing'  # change the backend for the generator (default is `multiprocessing`)
        >>> shac.concurrent_evaluators()  # change the backend of the evaluator to use `threading`
        ```

        Has an adaptive behaviour based on what epoch it is on, since later epochs require
        far more number of samples to generate a single batch of samples. When the epoch
        number increases beyond 10, it doubles the number of generator processes.

        This adaptivity can be removed by setting the parameter `limit_memory` to True.
        ```
        >>> shac.limit_memory = True
        ```

        # Arguments:
            evaluation_function ((tf.Session, int, list) -> float): The evaluation function is
            passed a managed Tensorflow Session, the integer id (of the worker) and the
            sampled hyper parameters in an OrderedDict. The evaluation function is expected
            to pass a python floating point number representing the evaluated value.
            skip_cv_checks (bool): If set, will not perform 5 fold cross validation check
                on the models before adding them to the classifer list. Useful when the
                batch size is small.
            early_stop (bool): Stop running if fail to find a classifier that beats the
                last stage of evaluations.
            relax_checks (bool): If set, will allow samples who do not pass all of the
                checks from all classifiers. Can be useful when large number of models
                are present and remaining search space is not big enough to allow sample
                to pass through all checks.
            callbacks (list | None): Optional list of callbacks that are executed when
                the engine is being trained. `History` callback is automatically added
                for all calls to `fit`.

        # Returns:
            A `History` object which tracks all the important information
            during training, and can be accessed using `history.history`
            as a dictionary.
        """
        return super(TensorflowSHAC, self).fit(eval_fn, skip_cv_checks=skip_cv_checks,
                                               early_stop=early_stop, relax_checks=relax_checks,
                                               callbacks=callbacks)
