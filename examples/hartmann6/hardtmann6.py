import os
import time
import numpy as np

from pyshac.config import hyperparameters as hp, data
from pyshac.core import engine

np.random.seed(0)


def get_hartmann6_hyperparameter_list():
    h = [hp.UniformContinuousHyperParameter('h%d' % i, 0.0, 1.0) for i in range(6)]
    return h


def evaluation_hartmann6(worker_id, params):
    """ Code ported from https://www.sfu.ca/~ssurjano/Code/hart6scm.html
    Global Minimum = -3.32237
    """
    alpha = np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float32)

    A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], dtype=np.float32)

    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]], dtype=np.float32)

    xx = np.array(list(params.values()), dtype=np.float32)

    xx = xx.reshape((6, 1))

    inner = np.sum(A * ((xx.T - P) ** 2), axis=-1)
    inner = np.exp(-inner)
    outer = np.sum(alpha * inner)

    return -outer


def check_hartmann6_impl():
    # Optimal _parameters
    x = [0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_hartmann6(0, params)

    assert np.allclose(loss, -3.32237)


def run_shac_hartmann6():
    total_budget = 200
    num_batches = 20
    objective = 'min'

    params = get_hartmann6_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(evaluation_hartmann6, h, total_budget=total_budget,
                       num_batches=num_batches, objective=objective)

    # do parallel work for fast processing
    shac.num_parallel_generators = 8
    shac.num_parallel_evaluators = 1

    print()

    # training
    if os.path.exists('shac/'):
        shac.restore_data()

    shac.fit(skip_cv_checks=True)

    print()
    print("Evaluating after training")
    predictions = shac.predict(num_batches=1, num_workers_per_batch=1)
    pred_evals = [evaluation_hartmann6(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Predicted mean : ", pred_mean)


if __name__ == '__main__':

    check_hartmann6_impl()
    # print('Time for 1000 iterations = ', timeit.timeit("check_hartmann6_impl()",
    #                                                    setup="from __main__ import check_hartmann6_impl",
    #                                                    number=1000))

    """ Train """
    # start = time.time()
    # run_shac_hartmann6()
    # end = time.time()
    # print("Time in seconds : ", end - start)

    """ Evaluation """
    shac = engine.SHAC(evaluation_hartmann6, None, total_budget=200,
                       num_batches=5, objective='min')

    shac.restore_data()

    start = time.time()
    predictions = shac.predict(1, num_workers_per_batch=5, max_classfiers=17)
    end = time.time()
    print("Time in seconds : ", end - start)

    pred_evals = [evaluation_hartmann6(0, pred) for pred in predictions]
    pred_mean = float(np.mean(pred_evals))
    pred_std = float(np.std(pred_evals))

    print()
    print("Predicted results : %0.5f +- (%0.5f)" % (pred_mean, pred_std))

"""
Results

Number of workers possible : 40
Using 40 parallel workers, it will require 5 epochs to fit 4 classifiers.
UserWarning: Number of workers exceeds 8 cores on device. Reducing parallel number of cores used to prevent resource starvation.
  "number of cores used to prevent resource starvation." % (cpu_count))

Found and restored dataset containing 180 samples
Found and restored 18 classifiers

Evaluating 1 batches (each containing 5 samples) with 5 generator (multiprocessing backend)
Number of classifiers availale = 17 (131072 samples generated per accepted sample on average)
[Parallel(n_jobs=5)]: Done   2 out of   5 | elapsed:  2.0min remaining:  3.1min
UserWarning: Could not find a sample after 131072 checks. You should consider using `relax_checks` to reduce this constraint or wait it out.
  "this constraint or wait it out." % (total_count))
UserWarning: Could not find a sample after 131072 checks. You should consider using `relax_checks` to reduce this constraint or wait it out.
  "this constraint or wait it out." % (total_count))
UserWarning: Could not find a sample after 131072 checks. You should consider using `relax_checks` to reduce this constraint or wait it out.
  "this constraint or wait it out." % (total_count))
[Parallel(n_jobs=5)]: Done   3 out of   5 | elapsed:  6.8min remaining:  4.5min
UserWarning: Could not find a sample after 262144 checks. You should consider using `relax_checks` to reduce this constraint or wait it out.
  "this constraint or wait it out." % (total_count))
[Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed: 12.6min remaining:    0.0s
[Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed: 12.6min finished
[Parallel(n_jobs=5)]: Done   2 out of   5 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=5)]: Done   3 out of   5 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed:    0.0s finished
Time in seconds :  759.0661308765411

Predicted results : -2.56789 +- (0.32437)

"""