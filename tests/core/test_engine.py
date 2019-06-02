import os
import shutil
import six
import pytest
import warnings
import numpy as np

from pyshac.config import hyperparameters as hp, data
from pyshac.core import engine
from pyshac.core.managed import torch_engine


warnings.simplefilter('ignore', DeprecationWarning)

# compatible with both Python 2 and 3
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


# wrapper function to clean up saved files and be deterministic
def optimizer_wrapper(func):
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        np.random.seed(0)
        output = func(*args, **kwargs)
        np.random.seed(None)

        # remove temporary files
        if os.path.exists('shac/'):
            shutil.rmtree('shac/')

        if os.path.exists('custom/'):
            shutil.rmtree('custom/')

        return output
    return wrapper


# wrapper function to clean up saved files and be deterministic
# when the optimizer is internally seeded
def seeded_optimizer_wrapper(func):
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)

        # remove temporary files
        if os.path.exists('shac/'):
            shutil.rmtree('shac/')

        if os.path.exists('custom/'):
            shutil.rmtree('custom/')

        return output
    return wrapper


def create_mock_dataset():
    np.random.seed(0)

    with open('shac/mock.csv', 'w') as f:
        f.write('id,x,y,scores\n')
        f.flush()

        template = '%d,%0.4f,%0.4f,%0.5f\n'

        for i in range(1000):
            x = np.random.uniform(-1., 1.)
            y = np.random.normal(0, 5.)
            score = x ** 2 + y ** 3
            f.write(template % (i, x, y, score))
            f.flush()


def get_hyperparameter_list():
    h1 = hp.DiscreteHyperParameter('h1', [0, 1, 2])
    h2 = hp.DiscreteHyperParameter('h2', [3, 4, 5, 6])
    h3 = hp.UniformContinuousHyperParameter('h3', 1, 25)
    h4 = hp.DiscreteHyperParameter('h4', ['v1', 'v2'])
    return [h1, h2, h3, h4]


def get_multi_parameter_list():
    h1 = hp.MultiDiscreteHyperParameter('h1', [0, 1, 2], sample_count=2)
    h2 = hp.MultiDiscreteHyperParameter('h2', [3, 4, 5, 6], sample_count=3)
    h3 = hp.MultiUniformContinuousHyperParameter('h3', 7, 10, sample_count=5)
    h4 = hp.MultiDiscreteHyperParameter('h4', ['v1', 'v2'], sample_count=4)
    return [h1, h2, h3, h4]


def get_hartmann6_hyperparameter_list():
    h = [hp.UniformContinuousHyperParameter('h%d' % i, 0.0, 1.0) for i in range(6)]
    return h


def get_branin_hyperparameter_list():
    h1 = hp.UniformContinuousHyperParameter('h1', -5.0, 10.0)
    h2 = hp.UniformContinuousHyperParameter('h2', 0.0, 15.0)
    return [h1, h2]


def evaluation_simple(worker_id, params):
    values = list(params.values())[:3]
    metric = np.sum(values)
    print('objective value =', metric)
    return metric


def evaluation_simple_multi(worker_id, params):
    values = data.flatten_parameters(params)[:10]
    metric = np.sum(values)
    print('objective value (multi params) =', metric)
    return metric


def evaluation_hartmann6(worker_id, params):
    """ Code ported from https://www.sfu.ca/~ssurjano/Code/hart6scm.html
    Global Minimum = -3.32237
    """

    xx = np.array(list(params.values()), dtype=np.float32)

    alpha = np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float32)

    A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], dtype=np.float32)

    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]], dtype=np.float64)

    xx = xx.reshape((6, 1))

    inner = np.sum(A * ((xx.T - P) ** 2), axis=-1)
    inner = np.exp(-inner)
    outer = np.sum(alpha * inner)

    return -outer


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


def test_hartmann6_impl():
    # Optimal _parameters
    x = [0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_hartmann6(0, params)
    assert np.allclose(loss, -3.32237)


def test_branin_impl():
    # Optimal parameter 1
    x = [-np.pi, 12.275]

    params = data.OrderedDict()
    for i, xx in enumerate(x):
        params['h%d' % i] = xx

    loss = evaluation_branin(0, params)
    print(loss)
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


@optimizer_wrapper
def test_shac_initialization():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    # direct params list submission
    shac = engine.SHAC(params, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    # submission of HyperParameterList
    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    # default number of parallel executors
    shac.set_num_parallel_generators(None)
    shac.set_num_parallel_evaluators(None)

    shac.concurrent_evaluators()
    shac.parallel_evaluators()

    assert shac.generator_backend == 'loky'
    assert shac.evaluator_backend == 'loky'

    shac.num_parallel_generators = 20
    assert shac.num_parallel_generators == 20

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    with pytest.raises(ValueError):
        shac.generator_backend = 'random'

    with pytest.raises(ValueError):
        shac.evaluator_backend = 'random'

    shac = engine.SHAC(None, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    # No parameters
    with pytest.raises(RuntimeError):
        shac.predict()


@optimizer_wrapper
def test_shac_simple():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    shac.set_seed(0)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 10
    assert shac.num_workers == 10
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 2
    shac.num_parallel_evaluators = 2

    print("Evaluating before training")

    # test prediction modes
    with pytest.raises(ValueError):
        shac.predict(max_classfiers=10)

    random_samples = shac.predict(num_samples=None, num_batches=None, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    assert len(random_eval) == 1

    random_samples = shac.predict(num_samples=4, num_batches=None, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    assert len(random_eval) == 4

    random_samples = shac.predict(num_samples=None, num_batches=1, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    assert len(random_eval) == 5

    random_samples = shac.predict(num_samples=2, num_batches=1, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    assert len(random_eval) == 7

    random_samples = shac.predict(num_batches=16, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    random_mean = np.mean(random_eval)

    print()

    # training
    shac.fit(evaluation_simple)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('shac/datasets/dataset.csv')
    assert os.path.exists('shac/classifiers/classifiers.pkl')

    print()
    print("Evaluating after training")
    predictions = shac.predict(num_batches=20, num_workers_per_batch=1)

    print("Shac preds", predictions)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = engine.SHAC(None, total_budget=total_budget,
                        num_batches=10, objective=objective)

    shac2.restore_data()

    shac2.set_seed(0)

    predictions = shac2.predict(num_batches=20, num_workers_per_batch=1)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # test no file found, yet no error
    shutil.rmtree('shac/')

    shac2.dataset = None
    shac2.classifiers = None
    shac2.restore_data()


@optimizer_wrapper
def test_shac_simple_custom_basepath():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective,
                       save_dir='custom')

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 10
    assert shac.num_workers == 10
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 2
    shac.num_parallel_evaluators = 2

    print("Evaluating before training")
    np.random.seed(0)

    # training
    shac.fit(evaluation_simple)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('custom/datasets/dataset.csv')
    assert os.path.exists('custom/classifiers/classifiers.pkl')

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = engine.SHAC(None, total_budget=total_budget,
                        num_batches=10, objective=objective,
                        save_dir='custom')

    shac2.restore_data()

    # test no file found, yet no error
    shutil.rmtree('custom/')

    shac2.dataset = None
    shac2.classifiers = None
    shac2.restore_data()


@seeded_optimizer_wrapper
def test_shac_simple_seeded_manually():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    # set the seed manually
    shac.set_seed(0)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 10
    assert shac.num_workers == 10
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 2
    shac.num_parallel_evaluators = 2

    print("Evaluating before training")

    # test prediction modes
    with pytest.raises(ValueError):
        shac.predict(max_classfiers=10)

    shac.set_seed(None)

    random_samples = shac.predict(num_samples=None, num_batches=None, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    assert len(random_eval) == 1

    random_samples = shac.predict(num_samples=4, num_batches=None, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    assert len(random_eval) == 4

    random_samples = shac.predict(num_samples=None, num_batches=1, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    assert len(random_eval) == 5

    random_samples = shac.predict(num_samples=2, num_batches=1, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    assert len(random_eval) == 7

    random_samples = shac.predict(num_batches=16, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    random_mean = np.mean(random_eval)

    print()

    shac.set_seed(0)

    # training
    shac.fit(evaluation_simple)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('shac/datasets/dataset.csv')
    assert os.path.exists('shac/classifiers/classifiers.pkl')

    print()
    print("Evaluating after training")
    shac.set_seed(None)
    predictions = shac.predict(num_batches=20, num_workers_per_batch=1)

    print("Shac preds", predictions)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = engine.SHAC(None, total_budget=total_budget,
                        num_batches=10, objective=objective)

    shac2.restore_data()

    with shac2.as_deterministic(1):
        predictions = shac2.predict(num_batches=20, num_workers_per_batch=1)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # Check if predictions are unique
    evals = {}
    for val in random_eval:
        if val in evals:
            evals[val] += 1
        else:
            evals[val] = 1

    print(evals)
    assert len(evals) > 1

    # Test if two predictions are same with two evals of same seed
    with shac2.as_deterministic(0):
        predictions = shac2.predict(num_batches=20, num_workers_per_batch=1)
        pred_evals1 = [evaluation_simple(0, pred) for pred in predictions]

    with shac2.as_deterministic(0):
        predictions = shac2.predict(num_batches=20, num_workers_per_batch=1)
        pred_evals2 = [evaluation_simple(0, pred) for pred in predictions]

    for p1, p2 in zip(pred_evals1, pred_evals2):
        assert p1 == p2

    # test no file found, yet no error
    shutil.rmtree('shac/')

    shac2.dataset = None
    shac2.classifiers = None
    shac2.restore_data()


@optimizer_wrapper
def test_shac_simple_multiparameter():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 10
    assert shac.num_workers == 10
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 2
    shac.num_parallel_evaluators = 2

    # training
    shac.fit(evaluation_simple_multi)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('shac/datasets/dataset.csv')
    assert os.path.exists('shac/classifiers/classifiers.pkl')

    print()
    print("Evaluating after training")
    np.random.seed(0)

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = engine.SHAC(None, total_budget=total_budget,
                        num_batches=10, objective=objective)

    shac2.restore_data()

    np.random.seed(0)
    # test no file found, yet no error
    shutil.rmtree('shac/')

    shac2.dataset = None
    shac2.classifiers = None
    shac2.restore_data()


@optimizer_wrapper
def test_shac_simple_relax_checks():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 10
    assert shac.num_workers == 10
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 1
    shac.num_parallel_evaluators = 1

    print("Evaluating before training")
    np.random.seed(0)
    random_samples = shac.predict(num_batches=16, num_workers_per_batch=1)  # random sample predictions
    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    random_mean = np.mean(random_eval)

    print()

    # training
    shac.fit(evaluation_simple, relax_checks=True)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('shac/datasets/dataset.csv')
    assert os.path.exists('shac/classifiers/classifiers.pkl')

    print()
    print("Evaluating after training")
    np.random.seed(0)
    predictions = shac.predict(num_batches=16, num_workers_per_batch=1)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = engine.SHAC(None, total_budget=total_budget,
                        num_batches=10, objective=objective)

    shac2.restore_data()

    np.random.seed(0)
    predictions = shac.predict(num_batches=10, num_workers_per_batch=1, relax_checks=True)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # test no file found, yet no error
    shutil.rmtree('shac/')

    shac2.dataset = None
    shac2.classifiers = None
    shac2.restore_data()


@optimizer_wrapper
def test_shac_simple_early_stop():
    total_budget = 100
    batch_size = 20
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 5
    assert shac.num_workers == 5
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 1
    shac.num_parallel_evaluators = 1

    # training (with failure)
    shac.fit(evaluation_simple, early_stop=True, skip_cv_checks=True)
    assert len(shac.classifiers) == 0


@optimizer_wrapper
def test_shac_simple_torch():
    total_budget = 100
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = torch_engine.TorchSHAC(h, total_budget=total_budget, max_gpu_evaluators=0,
                                  num_batches=batch_size, objective=objective, max_cpu_evaluators=1)

    shac.set_seed(0)

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 20
    assert shac.num_workers == 20
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 1
    shac.num_parallel_evaluators = 1

    print("Evaluating before training")
    np.random.seed(0)
    random_samples = shac.predict(num_batches=16, num_workers_per_batch=1)  # random sample predictions

    random_eval = [evaluation_simple(0, sample) for sample in random_samples]
    random_mean = np.mean(random_eval)

    print()

    # training
    shac.fit(evaluation_simple)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('shac/datasets/dataset.csv')
    assert os.path.exists('shac/classifiers/classifiers.pkl')

    print()
    print("Evaluating after training")
    np.random.seed(0)
    predictions = shac.predict(num_batches=16, num_workers_per_batch=1)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = torch_engine.TorchSHAC(None, total_budget=total_budget, max_gpu_evaluators=1,
                                   num_batches=10, objective=objective, max_cpu_evaluators=2)

    assert shac2.limit_memory is True

    shac2.restore_data()

    shac2.set_seed(0)

    np.random.seed(0)
    predictions = shac2.predict(num_batches=10, num_workers_per_batch=1)
    pred_evals = [evaluation_simple(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # test no file found, yet no error
    shutil.rmtree('shac/')

    shac2.classifiers = None
    shac2.dataset = None
    shac2.restore_data()


@optimizer_wrapper
def test_shac_simple_torch_custom_basepath():
    total_budget = 50
    batch_size = 5
    objective = 'max'

    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    shac = torch_engine.TorchSHAC(h, total_budget=total_budget, max_gpu_evaluators=0,
                                  num_batches=batch_size, objective=objective, max_cpu_evaluators=1,
                                  save_dir='custom')

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 10
    assert shac.num_workers == 10
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 1
    shac.num_parallel_evaluators = 1

    shac.generator_backend = 'loky'

    # training
    shac.fit(evaluation_simple)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('custom/datasets/dataset.csv')
    assert os.path.exists('custom/classifiers/classifiers.pkl')

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = torch_engine.TorchSHAC(None, total_budget=total_budget, max_gpu_evaluators=1,
                                   num_batches=10, objective=objective, max_cpu_evaluators=2,
                                   save_dir='custom')

    assert shac2.limit_memory is True

    shac2.restore_data()

    # test no file found, yet no error
    shutil.rmtree('custom/')

    shac2.classifiers = None
    shac2.dataset = None
    shac2.restore_data()


@optimizer_wrapper
def test_shac_fit_dataset():
    total_budget = 1000
    batch_size = 5
    objective = 'max'

    params = [hp.UniformHP('x', -1., 1.),
              hp.NormalHP('y', 0., 5.)]
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    # create the mock dataset
    create_mock_dataset()

    # Test wrong path
    with pytest.raises(FileNotFoundError):
        shac.fit_dataset('random.csv')

    # Test wrong engine configurations
    shac3 = engine.SHAC(h, 50000, num_batches=5)

    # Number of samples required is more than provided samples
    with pytest.raises(ValueError):
        shac3.fit_dataset('shac/mock.csv')

    # Test `None` parameters for engine
    shac5 = engine.SHAC(None, total_budget, batch_size)

    with pytest.raises(ValueError):
        shac5.fit_dataset('shac/mock.csv')

    # Wrong number of set params
    shac4 = engine.SHAC([params[0]], total_budget, batch_size)

    with pytest.raises(ValueError):
        shac4.fit_dataset('shac/mock.csv')

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 200
    assert shac.num_workers == 200
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 2
    shac.num_parallel_evaluators = 2

    print("Evaluating before training")
    np.random.seed(0)

    # training
    shac.fit_dataset('shac/mock.csv', presort=False)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('shac/datasets/dataset.csv')
    assert os.path.exists('shac/classifiers/classifiers.pkl')

    print()
    print("Evaluating after training")
    np.random.seed(0)
    predictions = shac.predict(num_batches=16, num_workers_per_batch=1)

    def eval_fn(id, pred):
        return pred['x'] ** 2 + pred['y'] ** 3

    pred_evals = [eval_fn(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    random_x = np.random.uniform(-1., 1., size=1000)
    random_y = np.random.normal(0., 5., size=1000)
    random_eval = random_x ** 2 + random_y ** 3
    random_mean = np.mean(random_eval)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = engine.SHAC(None, total_budget=total_budget,
                        num_batches=10, objective=objective)

    shac2.restore_data()

    np.random.seed(0)
    predictions = shac.predict(num_batches=10, num_workers_per_batch=1)
    pred_evals = [eval_fn(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # test no file found, yet no error
    shutil.rmtree('shac/')

    shac2.dataset = None
    shac2.classifiers = None
    shac2.restore_data()


@optimizer_wrapper
def test_shac_fit_dataset_presort():
    total_budget = 1000
    batch_size = 5
    objective = 'max'

    params = [hp.UniformHP('x', -1., 1.),
              hp.NormalHP('y', 0., 5.)]
    h = hp.HyperParameterList(params)

    shac = engine.SHAC(h, total_budget=total_budget,
                       num_batches=batch_size, objective=objective)

    # create the mock dataset
    create_mock_dataset()

    assert shac.total_classifiers == min(max(batch_size - 1, 1), 18)
    assert shac._per_classifier_budget == 200
    assert shac.num_workers == 200
    assert len(shac.classifiers) == 0
    assert len(shac.dataset) == 0

    # do sequential work for debugging
    shac.num_parallel_generators = 2
    shac.num_parallel_evaluators = 2

    print("Evaluating before training")
    np.random.seed(0)

    # training
    shac.fit_dataset('shac/mock.csv', presort=True, skip_cv_checks=True, early_stop=True)

    assert len(shac.classifiers) <= shac.total_classifiers
    assert os.path.exists('shac/datasets/dataset.csv')
    assert os.path.exists('shac/classifiers/classifiers.pkl')

    print()
    print("Evaluating after training")
    np.random.seed(0)
    predictions = shac.predict(num_batches=16, num_workers_per_batch=1)

    def eval_fn(id, pred):
        return pred['x'] ** 2 + pred['y'] ** 3

    pred_evals = [eval_fn(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    random_x = np.random.uniform(-1., 1., size=1000)
    random_y = np.random.normal(0., 5., size=1000)
    random_eval = random_x ** 2 + random_y ** 3
    random_mean = np.mean(random_eval)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # Serialization
    shac.save_data()

    # Restore with different batchsize
    shac2 = engine.SHAC(None, total_budget=total_budget,
                        num_batches=10, objective=objective)

    shac2.restore_data()

    np.random.seed(0)
    predictions = shac.predict(num_batches=10, num_workers_per_batch=1)
    pred_evals = [eval_fn(0, pred) for pred in predictions]
    pred_mean = np.mean(pred_evals)

    print()
    print("Random mean : ", random_mean)
    print("Predicted mean : ", pred_mean)

    assert random_mean < pred_mean

    # test no file found, yet no error
    shutil.rmtree('shac/')

    shac2.dataset = None
    shac2.classifiers = None
    shac2.restore_data()


if __name__ == '__main__':
    pytest.main([__file__])
