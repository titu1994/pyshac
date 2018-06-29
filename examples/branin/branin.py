import os
import time
import numpy as np

from pyshac.config import hyperparameters as hp, data
from pyshac.core import engine

np.random.seed(0)


def get_branin_hyperparameter_list():
    h1 = hp.UniformContinuousHyperParameter('h1', -5.0, 10.0)
    h2 = hp.UniformContinuousHyperParameter('h2', 0.0, 15.0)
    return [h1, h2]


def evaluation_branin(worker_id, params):
    """ Code ported from https://www.sfu.ca/~ssurjano/Code/braninm.html
    Global Minimum = -0.397887
    """

    xx = list(params.values())
    x1, x2 = xx[0], xx[1]

    a = 1.0
    b = 5.1 / (4 * (np.pi ** 2))
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    term1 = a * ((x2 - b * (x1 ** 2) + c * x1 - r) ** 2)
    term2 = s * (1.0 - t) * np.cos(x1)

    out = term1 + term2 + s
    return out


def check_branin_impl():
    # Optimal parameter 1
    x = [-np.pi, 12.275]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin(0, params)
    assert np.allclose(loss, 0.397887)

    # Optimal parameter 2
    x = [np.pi, 2.275]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin(0, params)
    assert np.allclose(loss, 0.397887)

    # Optimal parameter 3
    x = [9.42478, 2.475]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin(0, params)
    assert np.allclose(loss, 0.397887)


def check_shac_branin():
    total_budget = 200
    num_batches = 20
    objective = 'min'

    params = get_branin_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(evaluation_branin, h, total_budget=total_budget,
                       num_batches=num_batches, objective=objective)

    # do parallel work for fast processing
    shac.num_parallel_generators = 8
    shac.num_parallel_evaluators = 1

    shac.generator_backend = 'multiprocessing'
    shac.evaluator_backend = 'threading'

    print()

    # training
    if os.path.exists('shac/'):
        shac.restore_data()

    shac.fit(skip_cv_checks=True)

    print()
    print("Evaluating after training")
    predictions = shac.predict(num_batches=1, num_workers_per_batch=1)
    pred_evals = [evaluation_branin(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Predicted mean : ", pred_mean)


if __name__ == '__main__':

    check_branin_impl()
    # print('Time for 1000 iterations = ', timeit.timeit("check_branin_impl()",
    #                                                    setup="from __main__ import check_branin_impl",
    #                                                    number=1000))

    """ Train """
    # start = time.time()
    # check_shac_branin()
    # end = time.time()
    # print("Time in seconds : ", end - start)

    """ Evaluation """
    shac = engine.SHAC(evaluation_branin, None, total_budget=200,
                       num_batches=5, objective='min')

    shac.restore_data()

    # takes about 10 mins on 8 cores
    start = time.time()
    predictions = shac.predict(1, num_workers_per_batch=5, max_classfiers=17)
    end = time.time()
    print("Time in seconds : ", end - start)

    pred_evals = [evaluation_branin(0, pred) for pred in predictions]
    pred_mean = float(np.mean(pred_evals))
    pred_std = float(np.std(pred_evals))

    print()
    print("Predicted results : %0.5f +- (%0.5f)" % (pred_mean, pred_std))

"""
Results : 

Using 40 parallel workers, it will require 5 epochs to fit 4 classifiers.
UserWarning: Number of workers exceeds 8 cores on device. Reducing parallel number of cores used to prevent resource starvation.
  "number of cores used to prevent resource starvation." % (cpu_count))

Found and restored dataset containing 180 samples
Found and restored 18 classifiers

Evaluating 1 batches (each containing 5 samples) with 5 generator (multiprocessing backend)
Number of classifiers availale = 17 (131072 samples generated per accepted sample on average)
[Parallel(n_jobs=5)]: Done   2 out of   5 | elapsed:  1.0min remaining:  1.5min
[Parallel(n_jobs=5)]: Done   3 out of   5 | elapsed:  3.0min remaining:  2.0min
UserWarning: Could not find a sample after 131072 checks. You should consider using `relax_checks` to reduce this constraint or wait it out.
  "this constraint or wait it out." % (total_count))
UserWarning: Could not find a sample after 262144 checks. You should consider using `relax_checks` to reduce this constraint or wait it out.
  "this constraint or wait it out." % (total_count))
[Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed: 10.0min remaining:    0.0s
[Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed: 10.0min finished
[Parallel(n_jobs=5)]: Done   2 out of   5 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=5)]: Done   3 out of   5 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed:    0.0s finished
Time in seconds :  597.6038899421692

Predicted results : 0.39901 +- (0.00092)

"""