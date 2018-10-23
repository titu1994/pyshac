import multiprocessing
import os
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import pyshac.config.data as data
import pyshac.config.hyperparameters as hp
import pyshac.config.callbacks as cb
import pyshac.utils.xgb_utils as xgb_utils
import xgboost as xgb
from joblib import Parallel, delayed

warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)

# compatible with Python 2 and 3:
ABC = ABCMeta('ABC', (object,), {'__slots__': ()})

# compatible with both Python 2 and 3
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class _SHAC(ABC):
    """
    Abstract engine which performs most of the operations necessary for Sequential
    Halving and Classification algorithm to work.

    # Arguments:
        evaluation_function ((*) -> float): an evaluation function
            that will be called in parallel by different processes or threads to
            evaluate the model built by the provided parameters.
        hyperparameter_list (hp.HyperParameterList | None): A list of parameters
            (or a HyperParameterList) that are passed to define the search space.
            Can be None, so that it is loaded from disk instead.
        total_budget (int): `N`. Defines the total number of models to evaluate.
        num_batches (int): `M`. Defines the number of batches the work is distributed
            to. Must be set such that `total budget` is divisible by `batch size`.
        objective (str): Can be `max` or `min`. Whether to maximise
            the evaluation measure or minimize it.
        max_classifiers (int): Maximum number of classifiers that
            are trained. Default (18) is according to the paper.

    # References:
        - [Parallel Architecture and Hyperparameter Search via Successive Halving and Classification](https://arxiv.org/abs/1805.10255)

    # Raises:
        ValueError: If `total budget` is not divisible by `batch size`.

    """

    def __init__(self, hyperparameter_list, total_budget, num_batches,
                 objective='max', max_classifiers=18):
        if total_budget % num_batches != 0:
            raise ValueError("Number of epochs must be divisible by the batch size !")

        if hyperparameter_list is not None and (
                not isinstance(hyperparameter_list, hp.HyperParameterList)):
            hyperparameter_list = hp.HyperParameterList(hyperparameter_list)

        print("Number of workers possible : %d" % (total_budget // num_batches))

        self.parameters = hyperparameter_list
        self.objective = objective
        self._total_budget = total_budget  # N
        self.num_batches = num_batches  # M

        self._max_classifiers = max_classifiers
        self._num_workers = self.total_budget // num_batches  # W
        self._total_classifiers = min(max(num_batches - 1, 1), max_classifiers)  # K

        # serializable
        self.dataset = data.Dataset(hyperparameter_list)
        self.classifiers = []  # type: list(xgb.XGBClassifier)

        # training variables
        self._dataset_index = 0
        self._per_classifier_budget = int(self.num_workers * np.floor(total_budget / (
            float(self.num_workers * (self.total_classifiers + 1)))))  # Tc

        print("Using %d parallel workers, it will require %d epochs to fit %d classifiers.\n"
              "Each classifier will be provided %d samples to train per epoch." % (
                  self.num_workers, total_budget // self.num_workers,
                  self._total_classifiers, self._per_classifier_budget,
              ))

        # Compute how many threads and processes will be used
        self._compute_parallelism()

        # serialization paths
        self._prepare_dirs()

    @abstractmethod
    def _evaluation_handler(self, func, worker_id, parameter_dict):
        """
        Abstract method that is overriden by the subclasses. Useful to allow
        additional work to be done to manage the process that will evaluate
        the model generated using the given hyper parameters.

        # Arguments:
            func ((*) -> float): User defined evaluation function passed to the
                subclass so as to wrap and supply additional arguments if necessary.
            worker_id (int): The integer id of the process, ranging from
            [0, num_parallel_evaluator).
            parameter_dict (OrderedDict(str, int | float | str)): An OrderedDict of
                (name, value) pairs where the name and order is based on the declaration
                in the list of parameters passed to the constructor.
            batch_args (list | None): Optional arguments that a subclass can pass to all
                batches if necessary.

        # Returns:
             float.
        """
        raise NotImplementedError()

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
        >>> shac = SHAC(eval, params, total_budget=100, num_batches=10)

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
            evaluation_function ((*) -> float): The evaluation function is passed
                atleast the integer id (of the worker) and the sampled hyper parameters in
                an OrderedDict.  The evaluation function is expected to pass a python
                floating point number representing the evaluated value.
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
        num_epochs = self.total_budget // self.num_workers

        if skip_cv_checks:
            num_splits = 1
        else:
            num_splits = 5

        # Prepare callback list
        callback_list = cb.CallbackList(callbacks)
        callback_list.set_engine(self)

        print("Training with %d generator (%s backend) and %d evaluator threads (%s backend) "
              "with a batch size of %d" % (
                  self._num_parallel_generators, self._generator_backend,
                  self._num_parallel_evaluators, self._evaluator_backend,
                  self.num_workers,
              ))

        # if restarting training on older dataset, skip the older samples
        begin_run_index = len(self.dataset)

        callback_list.on_train_begin({'begin_run_index': begin_run_index})

        for run_index in range(begin_run_index, self.total_budget, self.num_workers):
            # initialize logs
            logs = {}

            current_epoch = (run_index // self.num_workers) + 1
            print("Beginning epoch %0.4d out of %0.4d" % (current_epoch, num_epochs))
            print("Number of classifiers availale = %d (%d samples generated per accepted "
                  "sample on average)" % (len(self.classifiers), 2 ** len(self.classifiers)))

            gen_backend = self.generator_backend
            prefer = 'processes' if gen_backend in ['loky', 'multiprocessing'] else 'threads'

            if current_epoch >= 10 and not self._limit_memory:
                # Use double the number of generators since a lot of samples will
                # be rejected at these stages
                generator_threads = 2 * self.num_parallel_generators
            else:
                generator_threads = self.num_parallel_generators

            # Log per classifier budget and number of generator threads
            logs['generator_threads'] = generator_threads
            logs['per_classifier_budget'] = self._per_classifier_budget
            callback_list.on_epoch_begin(current_epoch, logs)

            with Parallel(generator_threads, temp_folder=self.temp_dir,
                          backend=gen_backend, verbose=10, prefer=prefer) as parallel_generator:

                # parallel sample generation
                samples = parallel_generator(delayed(self._sample_parameters)(relax_checks)
                                             for _ in range(self.num_workers))

                params = parallel_generator(delayed(self.dataset.prepare_parameter)(smpl)
                                            for smpl in samples)

                print("Finished generating %d samples" % len(samples))

            with Parallel(self._num_parallel_evaluators, temp_folder=self.temp_dir,
                          backend=self._evaluator_backend, verbose=10, prefer=prefer) as parallel_evaluator:

                # Log the parameters sampled
                logs['parameters'] = params
                callback_list.on_evaluation_begin(params, logs)

                # parallel evaluation
                eval_values = parallel_evaluator(delayed(self._evaluation_handler)(eval_fn,
                                                                                   wid % self._num_parallel_evaluators,
                                                                                   param)
                                                 for wid, param in enumerate(params))

            # serial collection of results
            eval_values = list(eval_values)
            print("Finished evaluating %d samples" % len(eval_values))

            # Log the evaluated values of the provided parameters
            logs['device_ids'] = [wid % self._num_parallel_evaluators
                                  for wid in range(len(params))]
            logs['evaluations'] = eval_values
            callback_list.on_evaluation_ended(eval_values, logs)

            # serial adding of data to dataset for consistency
            for x, y in zip(samples, eval_values):
                self.dataset.add_sample(x, y)

            # Log the updated samples in the dataset and the dataset index
            logs['dataset_index'] = self._dataset_index
            callback_list.on_dataset_changed(self.dataset, logs)

            # get the dataset as numpy arrays
            x, y = self._get_dataset_samples()

            # Log the generated model
            logs['model'] = None

            if len(y) >= self._per_classifier_budget:
                # encode the dataset to prepare for training
                x, y = self._prepare_train_dataset(x, y)

                # train a classifier on the encoded dataset
                model = self._train_classifier(x, y, num_splits=num_splits)
                print("Finished training the %d-th classifier" % (len(self.classifiers) + 1))

                # Log the generated model
                logs['model'] = model

                # check if model failed to train for some reason
                if model is not None:
                    self.classifiers.append(model)

                else:
                    # the model can be None because of 2 reasons
                    # if maximum number of classifiers reached, then tell the user the reason.
                    if len(self.classifiers) == self.total_classifiers:
                        print("Cannot train any more models as maximum number of models has been reached.")

                    else:
                        # tell the user that the model failed the training procedure
                        print("\nCould not find a good classifier, therefore continuing to next epochs without "
                              "adding a model.\n ")

                        # if early stop flag was set, stop training.
                        if early_stop:
                            print("Since `early_stop` is set, stopping exeution and serializing immediately.")
                            break

                # save_dataset the dataset uptil now
                self.dataset.save_dataset()
                self._dataset_index += self._per_classifier_budget

            print("\n\nFinished training %4d out of %4d epochs" % (current_epoch, num_epochs))
            print()

            print("Serializing data and models")
            self.save_data()
            print()

            # Call at end of epoch of training.
            callback_list.on_epoch_end(current_epoch, logs)

        print("Finished training all models !")

        history = cb.get_history(callback_list)
        callback_list.on_train_end(history.history)

        return history

    def fit_dataset(self, dataset_path, skip_cv_checks=False, early_stop=False, presort=True,
                    callbacks=None):
        """
        Uses the provided dataset file to train the engine, instead of using
        the sequentual halving and classification algorithm directly. The data
        provided in the path must strictly follow the format of the dataset
        maintained by the engine.

        # Standard format of datasets:
            Each dataset csv file must contain an integer id column named "id"
            as its 1st column, followed by several columns describing the values
            taken by the hyper parameters, and the final column must be for
            the the objective criterion, and *must* be named "scores".

            The csv file *must* contain a header, following the above format.

        # Example:
                id,hp1,hp2,scores
                0,1,h1,1.0
                1,1,h2,0.2
                2,0,h1,0.0
                3,0,h3,0.5
                ...

        # Arguments:
            dataset_path (str): The full or relative path to a csv file
                containing the values of the dataset.
            skip_cv_checks (bool): If set, will not perform 5 fold cross
                validation check on the models before adding them to the
                classifer list. Useful when the batch size is small.
            early_stop (bool): Stop running if fail to find a classifier
                that beats the last stage of evaluations.
            presort (bool): Boolean flag to determine whether to sort
                the values of the dataset prior to loading. Ascending or
                descending sort is selected based on whether the engine
                is maximizing or minimizing the objective. It is preferable
                to set this always, to train better classifiers.
            callbacks (list | None): Optional list of callbacks that are executed when
                the engine is being trained. `History` callback is automatically added
                for all calls to `fit_dataset`.

        # Raises:
            ValueError: If the number of hyper parameters in the file
                are not the same as the number of hyper parameters
                that are available to the engine or if the number of
                samples in the provided dataset are less than the
                required number of samples by the engine.
            FileNotFoundError: If the dataset is not available at the
                provided filepath.

        # Returns:
            A `History` object which tracks all the important information
            during training, and can be accessed using `history.history`
            as a dictionary.
        """
        if self.parameters is None:
            raise ValueError("Parameter list cannot be `None` when training "
                             "via an external dataset.")

        num_epochs = self.total_budget // self.num_workers

        if skip_cv_checks:
            num_splits = 1
        else:
            num_splits = 5

        # Prepare callback list
        callback_list = cb.CallbackList(callbacks)
        callback_list.set_engine(self)

        # Reset the original dataset
        self._dataset_index = 0
        self.dataset.clear()

        # Load the dataset into memory
        self._rebuild_dataset(dataset_path, presort)
        callback_list.on_dataset_changed(self.dataset)

        X, Y = self.dataset.get_dataset()

        begin_run_index = 0

        callback_list.on_train_begin({'begin_run_index': 0})

        for run_index in range(begin_run_index, self.total_budget, self.num_workers):
            # initialize logs
            logs = {}

            current_epoch = (run_index // self.num_workers) + 1
            print("Beginning epoch %0.4d out of %0.4d" % (current_epoch, num_epochs))
            print("Number of classifiers availale = %d (%d samples generated per accepted "
                  "sample on average)" % (len(self.classifiers), 2 ** len(self.classifiers)))

            # Log per classifier budget
            logs['per_classifier_budget'] = self._per_classifier_budget
            callback_list.on_epoch_begin(current_epoch, logs)

            # Extract the samples from the dataset
            params = X[run_index: run_index + self.num_workers]
            print("Finished generating %d samples" % len(params))

            # Log the extracted parameters
            logs['parameters'] = params
            callback_list.on_evaluation_begin(params, logs)

            # Extract the evaluation results from the dataset
            eval_values = Y[run_index: run_index + self.num_workers]

            # serial collection of results
            eval_values = list(eval_values)
            print("Finished evaluating %d samples" % len(eval_values))

            # Log the evaluations from the provided parameters
            logs['evaluations'] = eval_values
            callback_list.on_evaluation_ended(eval_values, logs)

            # get the dataset as numpy arrays
            x = np.array(params, dtype=np.object)
            y = np.array(eval_values, dtype=np.object)

            # Log the model generated
            logs['model'] = None

            if len(y) >= self._per_classifier_budget:
                # encode the dataset to prepare for training
                x, y = self._prepare_train_dataset(x, y)

                # train a classifier on the encoded dataset
                model = self._train_classifier(x, y, num_splits=num_splits)
                print("Finished training the %d-th classifier" % (len(self.classifiers) + 1))

                # Log the model generated
                logs['model'] = model

                # check if model failed to train for some reason
                if model is not None:
                    self.classifiers.append(model)

                else:
                    # the model can be None because of 2 reasons
                    # if maximum number of classifiers reached, then tell the user the reason.
                    if len(self.classifiers) == self.total_classifiers:
                        print("Cannot train any more models as maximum number of models has been reached.")

                    else:
                        # tell the user that the model failed the training procedure
                        print("\nCould not find a good classifier, therefore continuing to next epochs without "
                              "adding a model.\n ")

                        # if early stop flag was set, stop training.
                        if early_stop:
                            print("Since `early_stop` is set, stopping exeution and serializing immediately.")
                            break

                # Update the internal dataaset index
                self._dataset_index += self._per_classifier_budget

            print("\n\nFinished training %4d out of %4d epochs" % (current_epoch, num_epochs))
            print()

            print("Serializing data and models")
            self.save_data()
            print()

            # Call after end of epoch of training
            callback_list.on_epoch_end(current_epoch, logs)

        print("Finished training all models !")

        history = cb.get_history(callback_list)
        callback_list.on_train_end(history.history)

        return history

    def predict(self, num_samples=None, num_batches=None, num_workers_per_batch=None, relax_checks=False,
                max_classfiers=None):
        """
        Using trained classifiers, sample the search space and predict which samples
        can successfully pass through the cascade of classifiers.

        When using a full cascade of 18 classifiers, a vast amount of time to sample
        a single sample.

        !!!note "Sample mode vs Batch mode"
            Parameters can be generated in either sample mode or batch mode or any combination
            of the two.

            `num_samples` is on a per sample basis (1 sample generated per count). Can be `None` or an int >= 0.
            `num_batches` is on a per batch basis (M samples generated per count). Can be `None` or an integer >= 0.

            The two are combined to produce a total number of samples which are provided in a
            list.

        # Arguments:
            num_samples (None | int): Number of samples to be generated.
            num_batches (None | int): Number of batches of samples to be generated.
            num_workers_per_batch (int): Determines how many parallel threads / processes
                are created to generate the batches. For small batches, it is best to use 1.
                If left as `None`, defaults to `num_parallel_generators`.
            relax_checks (bool): If set, will allow samples who do not pass all of the
                checks from all classifiers. Can be useful when large number of models
                are present and remaining search space is not big enough to allow sample
                to pass through all checks.
            max_classfiers (int | None): Number of classifiers to use for sampling.
                If set to None, will use all classifiers.

        # Raises:
            ValueError: If `max_classifiers` is larger than the number of available
                classifiers.

        # Returns:
            batches of samples in the form of an OrderedDict
        """
        if max_classfiers is not None and max_classfiers > len(self.classifiers):
            raise ValueError("Maximum number of classifiers (%d) must be less than the number of "
                             "classifiers (%d)" % (max_classfiers, len(self.classifiers)))

        if num_samples is None and num_batches is None:
            sample_count = 1
        elif num_samples is None and num_batches is not None:
            sample_count = num_batches * self.num_batches
        elif num_samples is not None and num_batches is None:
            sample_count = num_samples
        else:
            sample_count = num_batches * self.num_batches + num_samples

        if max_classfiers is None:
            max_classfiers = len(self.classifiers)

        generator_threads = self._num_parallel_generators if num_workers_per_batch is None else num_workers_per_batch

        num_samples = sample_count // generator_threads + int(sample_count % generator_threads != 0)

        if len(self.classifiers) >= 10 and not self.limit_memory:
            # Use double the number of generators since a lot of samples will
            # be rejected at these stages
            generator_threads = 2 * generator_threads if num_workers_per_batch is None else num_workers_per_batch

        gen_backend = self.generator_backend
        prefer = 'processes' if gen_backend in ['loky', 'multiprocessing'] else 'threads'

        print("Evaluating %d batches (for a total of %d samples) with %d generator (%s backend)" % (
            num_samples, sample_count, generator_threads, gen_backend))

        print("Number of classifiers availale = %d (%d samples generated per accepted "
              "sample on average)" % (max_classfiers, 2 ** max_classfiers))

        with Parallel(generator_threads, temp_folder=self.temp_dir, verbose=10,
                      backend=gen_backend, prefer=prefer) as parallel_generator:

            sample_list = []
            for run_index in range(0, sample_count, generator_threads):
                count = min(sample_count - run_index, generator_threads)

                samples = parallel_generator(delayed(self._sample_parameters)(relax_checks,
                                                                              max_classfiers)
                                             for _ in range(count))

                params = parallel_generator(delayed(self.dataset.prepare_parameter)(smpl)
                                            for smpl in samples)

                sample_list.extend(params)

        return sample_list

    def _sample_parameters(self, relax_checks=False, max_classifiers=None):
        """
        Samples the underlying hyper parameters, checks if the sample passes through all
        of the classifiers, and only then submits it for evaluation.

        This is a very expensive process at large number of classifiers, as on average, it
        requires `2 ^ num_classifiers` number of samples to get a single sample to pass
        the cascade of classifier tests. At 18 classifiers, this is roughly 240k samples,
        and may be several times more than this if outliers exists.

        This also modifies the numpy random state before it begins sampling. This is
        primarily because it is pointless to have multiple processors sampling using the
        same random seed.

        # Arguments:
            relax_checks (bool): If set, will allow samples who do not pass all of the
                checks from all classifiers. Can be useful when large number of models
                are present and remaining search space is not big enough to allow sample
                to pass through all checks.
            max_classifiers (int | None): Number of classifiers to use for sampling.
                If set to None, will use all classifiers.

        # Returns:
            List of encoded sample value
        """
        np.random.RandomState(None)

        # If there are no classifiers, simply sample and pass through.
        if len(self.classifiers) == 0:
            sample = self.parameters.sample()
        else:
            # get the first sample and encode it.
            sample = self.parameters.sample()
            sample, _ = self.dataset.encode_dataset([sample], objective=self.objective)

            # if we limit the number of classifiers, during `predict`, it is faster.
            # Not used during training.
            if max_classifiers is None:
                available_clfs = self.classifiers
            else:
                available_clfs = self.classifiers[:max_classifiers]

            # compute the average number of samples needed for a single sample to
            # pass through the cascade of classifiers
            average_num = 2 ** len(available_clfs)
            max_checks = int(average_num)

            counter = 0
            checks_relaxation_counter = 0
            total_count = 0

            clf_count = len(available_clfs)

            # keep sampling and testing until a sample passes all checks
            while not xgb_utils.evaluate_models(sample, available_clfs, checks_relaxation_counter):
                sample = self.parameters.sample()
                sample, _ = self.dataset.encode_dataset([sample], objective=self.objective)
                counter += 1

                # notify users at intervals which are multiples of the average number
                # of samples.
                if counter >= max_checks:
                    total_count += max_checks

                    # If checks are relaxed, uses fewer classifiers approvals in next stage
                    if relax_checks:
                        checks_relaxation_counter += 1

                        warnings.warn("Relaxing check to pass %d classifiers only" % (
                                clf_count - checks_relaxation_counter))

                    else:
                        # Otherwise, simply notify user that we could not find a sample
                        warnings.warn("Could not find a sample after %d checks. "
                                      "You should consider using `relax_checks` to reduce "
                                      "this constraint or wait it out." % (total_count))

                    counter = 0
                else:
                    counter += 1

            sample = sample[0].tolist()
            sample = self.dataset.decode_dataset([sample])
            sample = sample[0].tolist()

        return sample

    def _rebuild_dataset(self, dataset_path, presort):
        """
        Uses the provided path to load the values contained in the dataset
        in a standard format as shown below, and then reloads these values
        into the internal dataset of the engine.

        Standard format of datasets:
        Each dataset csv file must contain an integer id column named "id"
        as its 1st column, followed by several columns describing the values
        taken by the hyper parameters, and the final column must be for
        the the objective criterion, and *must* be named "scores".

        The csv file *must* contain a header, following the above format.

        Example:
            id,hp1,hp2,scores
            0,1,1,1.0
            1.0,1,0.0
            2,1,0,0.0
            3,0,0,1.0
            ...

        # Arguments:
            dataset_path (str): The full or relative path to a csv file
                containing the values of the dataset.
            presort (bool): Boolean flag to determine whether to sort
                the values of the dataset prior to loading. Ascending or
                descending sort is selected based on whether the engine
                is maximizing or minimizing the objective.

        # Raises:
            ValueError: If the number of hyper parameters in the file
                are not the same as the number of hyper parameters
                that are available to the engine or if the number of
                samples in the provided dataset are less than the
                required number of samples by the engine.
            FileNotFoundError: If the dataset is not available at the
                provided filepath.

        # Returns:
            A re-constructed dataset that has been restored with provided values.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("The dataset at provided path %s was not "
                                    "found. Please provide the correct "
                                    "path." % dataset_path)

        print("Deserializing dataset...")
        df = pd.read_csv(dataset_path, header=0, encoding='utf-8')

        if len(df) < self._total_budget:
            raise ValueError("Number of available samples from provided dataset (%d) "
                             "is less than the required number of samples (%d)" % (
                                len(df), self._total_budget,
                             ))

        cols = df.columns.values.tolist()
        df.drop(cols[0], axis=1, inplace=True)

        hyperparam_cols = cols[1:-1]
        if len(hyperparam_cols) != len(self.parameters):
            raise ValueError("Number of hyper parameters in provided dataset (%d)"
                             "does not match the number of hyper parameters "
                             "available for the engine (%d)." % (
                                 len(hyperparam_cols), len(self.parameters)
                             ))

        if presort:
            if self.objective == 'max':
                sort_ascending = True
            else:
                sort_ascending = False

            df.sort_values('scores', ascending=sort_ascending, inplace=True)

        x = df[cols[1:-1]].values
        y = df[cols[-1]].values

        # Load the values into the dataset
        self.dataset.set_dataset(x.tolist(), y.tolist())

        # Save the dataset alongside the parameters
        self.dataset.save_dataset()

        return self.dataset

    def _prepare_dirs(self):
        """
        Creates directorues where the temporary files from joblib and the trained
        classifiers will be saved.
        """
        self.temp_dir = os.path.join('shac', 'temp')
        self.clf_dir = os.path.join('shac', 'classifiers')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        if not os.path.exists(self.clf_dir):
            os.makedirs(self.clf_dir)

    def _compute_parallelism(self):
        """
        Compute the number of threads / processes allocated to the generators and the
        evaluators.
        """
        cpu_count = multiprocessing.cpu_count()
        if self.num_workers > cpu_count:
            warnings.warn("Number of workers exceeds %d cores on device. Reducing parallel "
                          "number of cores used to prevent resource starvation." % (cpu_count))

            self._num_parallel_generators = max(cpu_count, 1)
            self._num_parallel_evaluators = max(cpu_count, 1)
        else:
            self._num_parallel_generators = max(self.num_workers, 1)
            self._num_parallel_evaluators = max(self.num_workers, 1)

        self._generator_backend = 'loky'
        self._evaluator_backend = 'loky'
        self._limit_memory = False

    def _get_dataset_samples(self):
        """
        Packs the samples into a numpy array, and then retrieves only the current batch
        of samples using the `dataset_index`/

        # Returns:
             a tuple of (np.ndarray, np.ndarray) representing the training set X and the
             evaluations Y
        """
        x, y = self.dataset.get_dataset()
        x = x[self._dataset_index: self._dataset_index + self._per_classifier_budget]
        y = y[self._dataset_index: self._dataset_index + self._per_classifier_budget]

        return x, y

    def _prepare_train_dataset(self, x, y):
        """
        Encodes all of the samples from the dataset to be passed to the classifiers
        for training.

        # Arguments:
            x (np.ndarray): A numpy array representing all of the samples that will
                be used for training the classifiers.
            y (np.ndarray): A numpy array representing all of the evaluations that will
                be used for training the classifiers.

        # Returns:
             a tuple of (np.ndarray, np.ndarray) representing the encoded dataset.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        x, y = self.dataset.encode_dataset(x, y, objective=self.objective)
        return x, y

    def _train_classifier(self, X, Y, num_splits=5):
        """
        Trains a classifier if we have less than `total_classifiers` number of models,
        else simply returns.

        # Arguments:
            X (np.ndarray): A numpy array representing all of the encoded samples that
                 will be used for training the classifiers.
            Y (np.ndarray): A numpy array representing all of the encoded evaluations
                 that will be used for training the classifiers.
            num_splits (int): number of splits to perform cross validated training.
                Useful if we have sufficient samples for training a cross validated
                model per batch, else set to 1 (do not cross validate).

        # Returns:
            A trained model or None
        """
        if len(self.classifiers) < self.total_classifiers:
            model = xgb_utils.train_single_model(X, Y, num_splits=num_splits,
                                                 n_jobs=self.num_workers)
            return model
        else:
            return None

    def save_data(self):
        """
        Serialize the class objects by serializing the dataset and the trained models.
        """
        # save_dataset dataset
        self.dataset.save_dataset()

        # save_dataset models
        xgb_utils.save_classifiers(self.classifiers, self.clf_dir)

    def restore_data(self):
        """
        Recover the serialized class objects by loading the dataset and the trained models
        from the default save directories.
        """
        try:
            self.dataset = data.Dataset.load_from_directory()
        except FileNotFoundError:
            pass

        try:
            self.classifiers = xgb_utils.restore_classifiers(self.clf_dir)
        except FileNotFoundError:
            pass

        self.parameters = self.dataset.parameters
        self._dataset_index = len(self.dataset)

        if len(self.dataset) > 0:
            print("\nFound and restored dataset containing %d samples" % (len(self.dataset)))

        if len(self.classifiers) > 0:
            print("Found and restored %d classifiers" % (len(self.classifiers)))

        print()

    def parallel_evaluators(self):
        """
        Sets the evaluators to use the `loky` backend.

        The user must take the responsibility of thread safety and memory management.
        """
        print("Evaluators will now use the `loky` backend")
        self.evaluator_backend = 'loky'

    def concurrent_evaluators(self):
        """
        Sets the evaluators to use the threading backend, and therefore
        be locked by Python's GIL.

        While technically still "parallel", it is in fact concurrent execution
        rather than parallel execution of the evaluators.
        """
        print("Evaluators will now use the `threading` backend")
        self.evaluator_backend = 'threading'

    @property
    def total_budget(self):
        return self._total_budget

    @property
    def num_workers(self):
        return self._num_workers

    @property
    def total_classifiers(self):
        return self._total_classifiers

    @property
    def num_parallel_generators(self):
        return self._num_parallel_generators

    @num_parallel_generators.setter
    def num_parallel_generators(self, val):
        """


        # Arguments:

        """
        if val is None:
            cpu_count = multiprocessing.cpu_count()

            if self.num_workers > cpu_count:
                warnings.warn("Number of workers exceeds %d cores on device. Reducing parallel "
                              "number of cores used to prevent resource starvation." % (cpu_count))

                self._num_parallel_generators = max(cpu_count, 1)
            else:
                self._num_parallel_generators = max(self.num_workers, 1)
        else:
            self._num_parallel_generators = val

    def set_num_parallel_generators(self, n):
        """
        Check and sets the number of parallel generators. If None, checks if the number
        of workers exceeds the number of virtual cores. If it does, it warns about it
        and sets the max parallel generators to be the number of cores.

        # Arguments:
            n (int | None): The number of parallel generators required.
        """
        self.num_parallel_generators = n

    @property
    def num_parallel_evaluators(self):
        return self._num_parallel_evaluators

    @num_parallel_evaluators.setter
    def num_parallel_evaluators(self, val):
        if val is None:
            cpu_count = multiprocessing.cpu_count()

            if self.num_workers > cpu_count:
                warnings.warn("Number of workers exceeds %d cores on device. Reducing parallel "
                              "number of cores used to prevent resource starvation." % (cpu_count))

                self._num_parallel_evaluators = max(cpu_count, 1)
            else:
                self._num_parallel_evaluators = max(self.num_workers, 1)
        else:
            self._num_parallel_evaluators = val

    def set_num_parallel_evaluators(self, n):
        """
        Check and sets the number of parallel evaluators. If None, checks if the number
        of workers exceeds the number of virtual cores. If it does, it warns about it
        and sets the max parallel generators to be the number of cores.

        # Arguments:
            n (int | None): The number of parallel evaluators required.
        """
        self.num_parallel_evaluators = n

    @property
    def generator_backend(self):
        return self._generator_backend

    @generator_backend.setter
    def generator_backend(self, val):
        """
        Sets the backend of the generators. It is preferred to keep this as `loky`,
        since the number of samples required for larger number of classifiers is enormous, and
        processes are not limited by Python's Global Interpreter Lock (an issue with the `threading`
        backend).

        # Arguments:
            val (str): The backend to be assigned. Can be one of 'loky', 'multiprocessing' or
                'threading'.

        # Raises:
            ValueError: If the backend passed was anything other than 'multiprocessing' or
                'threading'
        """
        if val not in ['threading', 'multiprocessing', 'loky']:
            raise ValueError("This can be one of ['multiprocessing', 'threading']")

        self._generator_backend = val
        print("Generator backend set to %s" % val)

    @property
    def evaluator_backend(self):
        return self._evaluator_backend

    @evaluator_backend.setter
    def evaluator_backend(self, val):
        """
        Sets the backend of the evaluators. It is preferred to keep this as `loky`,
        since the time taken to evaluate a large number of samples for models can be large, and
        processes are not limited by Python's Global Interpreter Lock (an issue with the `threading`
        backend).

        # Arguments:
            val (str): The backend to be assigned. Can be one of 'loky', 'multiprocessing' or
                'threading'.

        # Raises:
            ValueError: If the backend passed was anything other than 'multiprocessing' or
                'threading'
        """
        if val not in ['threading', 'multiprocessing', 'loky']:
            raise ValueError("This can be one of ['multiprocessing', 'threading']")

        self._evaluator_backend = val
        print("Evaluator backend set to %s" % val)

    @property
    def limit_memory(self):
        return self._limit_memory

    @limit_memory.setter
    def limit_memory(self, limit):
        self._limit_memory = limit


class SHAC(_SHAC):
    """
    The default and generic implementation of the SHAC algorithm. It is a wrapper over the
    abstract class, and performs no additional maintenance over the evaluation function.

    It is fastest engine, but assumes that the evaluation function is thread safe and
    the system has sufficient memory to run several copies of the evaluation function
    at the same time.

    This engine is suitable for Numpy / PyTorch based work. Both numpy and PyTorch
    dynamically allocate memory, and therefore, as long as the system has enough memory
    to run multiple copies of the evaluation model, there is no additional memory management
    to be done.

    Still, for PyTorch, the number of evaluation processes should be carefully set,
    so as not to exhaust all CPU / GPU memory during execution.

    # Arguments:
        evaluation_function ((int, list) -> float): The evaluation function is passed
            only the integer id (of the worker) and the sampled hyper parameters in
            an OrderedDict.  The evaluation function is expected to pass a python
            floating point number representing the evaluated value.
        hyperparameter_list (hp.HyperParameterList | None): A list of parameters
            (or a HyperParameterList) that are passed to define the search space.
            Can be None, so that it is loaded from disk instead.
        total_budget (int): `N`. Defines the total number of models to evaluate.
        num_batches (int): `M`. Defines the number of batches the work is distributed
            to. Must be set such that `total budget` is divisible by `batch size`.
        objective (str): Can be `max` or `min`. Whether to maximise
            the evaluation measure or minimize it.
        max_classifiers (int): Maximum number of classifiers that
            are trained. Default (18) is according to the paper.

    # References:
        - [Parallel Architecture and Hyperparameter Search via Successive Halving and Classification](https://arxiv.org/abs/1805.10255)

    # Raises:
        ValueError: If `total budget` is not divisible by `batch size`.
    """

    def __init__(self, hyperparameter_list, total_budget, num_batches,
                 objective='max', max_classifiers=18):
        super(SHAC, self).__init__(hyperparameter_list, total_budget,
                                   num_batches=num_batches, objective=objective,
                                   max_classifiers=max_classifiers)

    def _evaluation_handler(self, func, worker_id, parameter_dict, *batch_args):
        """
        Basic implementation of the abstract handler. Performs no additional actions,
        and simply evaluates the user model. It assumes the user's evaluation function
        is thread / process safe.

        # Arguments:
            func ((int, list) -> float): The evaluation function is passed
                only the integer id (of the worker) and the sampled hyper parameters in
                an OrderedDict.  The evaluation function is expected to pass a python
                floating point number representing the evaluated value.
            worker_id (int): The integer id of the process, ranging from
                [0, num_parallel_evaluator).
            parameter_dict (OrderedDict(str, int | float | str)): An OrderedDict of
                (name, value) pairs where the name and order is based on the declaration
                in the list of parameters passed to the constructor.
            batch_args (list | None): Optional arguments that a subclass can pass to all
                batches if necessary.

        # Returns:
             float representing the evaluated value.
        """
        output = func(worker_id, parameter_dict)
        output = float(output)
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
        >>> shac = SHAC(params, total_budget=100, num_batches=10)

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
            evaluation_function ((int, list) -> float): The evaluation function is passed
                only the integer id (of the worker) and the sampled hyper parameters in
                an OrderedDict.  The evaluation function is expected to pass a python
                floating point number representing the evaluated value.
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
        return super(SHAC, self).fit(eval_fn, skip_cv_checks=skip_cv_checks,
                                     early_stop=early_stop, relax_checks=relax_checks,
                                     callbacks=callbacks)
