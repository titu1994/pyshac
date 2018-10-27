import os
import shutil
import six
import pytest
import numpy as np

from pyshac.config import hyperparameters as hp, data


# compatible with both Python 2 and 3
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def deterministic_test(func):
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        np.random.seed(0)
        output = func(*args, **kwargs)
        np.random.seed(None)

        return output
    return wrapper


# wrapper function to clean up saved files
def cleanup_dirs(func):
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


def get_hyperparameter_list():
    h1 = hp.DiscreteHyperParameter('h1', [0, 1, 2])
    h2 = hp.DiscreteHyperParameter('h2', [3, 4, 5, 6])
    h3 = hp.UniformContinuousHyperParameter('h3', 7, 10)
    h4 = hp.DiscreteHyperParameter('h4', ['v1', 'v2'])
    return [h1, h2, h3, h4]


def get_multi_parameter_list():
    h1 = hp.MultiDiscreteHyperParameter('h1', [0, 1, 2], sample_count=2)
    h2 = hp.MultiDiscreteHyperParameter('h2', [3, 4, 5, 6], sample_count=3)
    h3 = hp.MultiUniformContinuousHyperParameter('h3', 7, 10, sample_count=5)
    h4 = hp.MultiDiscreteHyperParameter('h4', ['v1', 'v2'], sample_count=4)
    return [h1, h2, h3, h4]


@cleanup_dirs
def test_dataset_param_list():
    params = get_hyperparameter_list()

    dataset = data.Dataset(params)
    assert isinstance(dataset._parameters, hp.HyperParameterList)

    dataset.set_parameters(params)
    assert isinstance(dataset._parameters, hp.HyperParameterList)

    h = hp.HyperParameterList(params)
    dataset.set_parameters(h)
    assert isinstance(dataset._parameters, hp.HyperParameterList)


@cleanup_dirs
def test_dataset_multi_param_list():
    params = get_multi_parameter_list()

    dataset = data.Dataset(params)
    assert isinstance(dataset._parameters, hp.HyperParameterList)

    dataset.set_parameters(params)
    assert isinstance(dataset._parameters, hp.HyperParameterList)

    h = hp.HyperParameterList(params)
    dataset.set_parameters(h)
    assert isinstance(dataset._parameters, hp.HyperParameterList)


@cleanup_dirs
def test_dataset_basedir():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)
    assert os.path.exists(dataset.basedir)


@cleanup_dirs
def test_dataset_basedir_custom():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h, basedir='custom')
    assert os.path.exists(dataset.basedir)
    assert not os.path.exists('shac')


@cleanup_dirs
def test_dataset_add_sample():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    x, y = dataset.get_dataset()
    assert len(dataset) == 5
    assert x.shape == (5, 4)
    assert y.shape == (5,)


@cleanup_dirs
def test_dataset_multi_add_sample():
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    x, y = dataset.get_dataset()
    assert len(dataset) == 5
    assert x.shape == (5, 14)
    assert y.shape == (5,)


@cleanup_dirs
def test_set_dataset():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)
    # numpy arrays
    samples = [(np.array(h.sample()), np.random.uniform()) for _ in range(5)]

    x, y = zip(*samples)
    x = np.array(x)
    y = np.array(y)
    dataset.set_dataset(x, y)
    assert len(dataset) == 5

    dataset.clear()

    # python arrays
    samples = [(h.sample(), float(np.random.uniform())) for _ in range(5)]

    x, y = zip(*samples)
    dataset.set_dataset(x, y)
    assert len(dataset) == 5

    # None data
    with pytest.raises(TypeError):
        dataset.set_dataset(None, int(6))

    with pytest.raises(TypeError):
        dataset.set_dataset([1, 2, 3], None)

    with pytest.raises(TypeError):
        dataset.set_dataset(None, None)


@cleanup_dirs
def test_multi_set_dataset():
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)
    # numpy arrays
    samples = [(np.array(h.sample()), np.random.uniform()) for _ in range(5)]

    x, y = zip(*samples)
    x = np.array(x)
    y = np.array(y)
    dataset.set_dataset(x, y)
    assert len(dataset) == 5

    dataset.clear()

    # python arrays
    samples = [(h.sample(), float(np.random.uniform())) for _ in range(5)]

    x, y = zip(*samples)
    dataset.set_dataset(x, y)
    assert len(dataset) == 5

    # None data
    with pytest.raises(TypeError):
        dataset.set_dataset(None, int(6))

    with pytest.raises(TypeError):
        dataset.set_dataset([1, 2, 3], None)

    with pytest.raises(TypeError):
        dataset.set_dataset(None, None)


@cleanup_dirs
@deterministic_test
def test_dataset_get_best_parameters():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    with pytest.raises(ValueError):
        dataset.get_best_parameters(None)

    # Test with empty dataset
    assert dataset.get_best_parameters() is None

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    objective_values = [v for h, v in samples]
    min_index = np.argmin(objective_values)
    max_index = np.argmax(objective_values)

    max_hp = list(dataset.get_best_parameters(objective='max').values())
    min_hp = list(dataset.get_best_parameters(objective='min').values())

    assert max_hp == samples[max_index][0]
    assert min_hp == samples[min_index][0]


@cleanup_dirs
@deterministic_test
def test_dataset_multi_get_best_parameters():
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    with pytest.raises(ValueError):
        dataset.get_best_parameters(None)

    # Test with empty dataset
    assert dataset.get_best_parameters() is None

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]

    for sample in samples:
        dataset.add_sample(*sample)

    objective_values = [v for h, v in samples]
    min_index = np.argmin(objective_values)
    max_index = np.argmax(objective_values)

    max_hp = data.flatten_parameters(dataset.get_best_parameters(objective='max'))
    min_hp = data.flatten_parameters(dataset.get_best_parameters(objective='min'))

    assert max_hp == samples[max_index][0]
    assert min_hp == samples[min_index][0]


@cleanup_dirs
def test_dataset_parameters():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)
    assert len(params) == len(dataset.parameters)

    dataset.parameters = params
    assert len(params) == len(dataset.parameters)


@cleanup_dirs
def test_dataset_serialization_deserialization():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    # serialization
    dataset.save_dataset()

    assert len(dataset) == 5
    assert os.path.exists(dataset.data_path)
    assert os.path.exists(dataset.parameter_path)

    # deserialization
    dataset.clear()
    assert len(dataset) == 0

    dataset.restore_dataset()

    assert len(dataset) == 5
    assert os.path.exists(dataset.data_path)
    assert os.path.exists(dataset.parameter_path)

    # deserialization from class
    path = os.path.join('shac', 'datasets')
    dataset2 = data.Dataset.load_from_directory(path)

    assert dataset2.parameters is not None
    assert len(dataset2.X) == 5
    assert len(dataset2.Y) == 5
    assert len(dataset2) == 5

    dataset3 = data.Dataset.load_from_directory()

    assert dataset3.parameters is not None
    assert len(dataset3.X) == 5
    assert len(dataset3.Y) == 5

    # serialization of empty get_dataset
    dataset = data.Dataset()

    with pytest.raises(FileNotFoundError):
        dataset.load_from_directory('null')

    with pytest.raises(ValueError):
        dataset.save_dataset()


@cleanup_dirs
def test_dataset_multi_serialization_deserialization():
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    # serialization
    dataset.save_dataset()

    assert len(dataset) == 5
    assert os.path.exists(dataset.data_path)
    assert os.path.exists(dataset.parameter_path)

    # deserialization
    dataset.clear()
    assert len(dataset) == 0

    dataset.restore_dataset()

    assert len(dataset) == 5
    assert os.path.exists(dataset.data_path)
    assert os.path.exists(dataset.parameter_path)

    # deserialization from class
    path = os.path.join('shac', 'datasets')
    dataset2 = data.Dataset.load_from_directory(path)

    assert dataset2.parameters is not None
    assert len(dataset2.X) == 5
    assert len(dataset2.Y) == 5
    assert len(dataset2) == 5

    dataset3 = data.Dataset.load_from_directory()

    assert dataset3.parameters is not None
    assert len(dataset3.X) == 5
    assert len(dataset3.Y) == 5

    # serialization of empty get_dataset
    dataset = data.Dataset()

    with pytest.raises(FileNotFoundError):
        dataset.load_from_directory('null')

    with pytest.raises(ValueError):
        dataset.save_dataset()


@cleanup_dirs
def test_dataset_serialization_deserialization_custom_basepath():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h, basedir='custom')

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    # serialization
    dataset.save_dataset()

    assert len(dataset) == 5
    assert os.path.exists(dataset.data_path)
    assert os.path.exists(dataset.parameter_path)

    # deserialization
    dataset.clear()
    assert len(dataset) == 0

    dataset.restore_dataset()

    assert len(dataset) == 5
    assert os.path.exists(dataset.data_path)
    assert os.path.exists(dataset.parameter_path)

    # deserialization from class
    path = os.path.join('custom', 'datasets')
    dataset2 = data.Dataset.load_from_directory(path)

    assert dataset2.parameters is not None
    assert len(dataset2.X) == 5
    assert len(dataset2.Y) == 5
    assert len(dataset2) == 5

    dataset3 = data.Dataset.load_from_directory('custom')

    assert dataset3.parameters is not None
    assert len(dataset3.X) == 5
    assert len(dataset3.Y) == 5

    # serialization of empty get_dataset
    dataset = data.Dataset(basedir='custom')

    with pytest.raises(FileNotFoundError):
        dataset.load_from_directory('null')

    with pytest.raises(ValueError):
        dataset.save_dataset()


@cleanup_dirs
def test_dataset_serialization_deserialization_custom_param():
    class MockDiscreteHyperParameter(hp.DiscreteHyperParameter):

        def __init__(self, name, values):
            super(MockDiscreteHyperParameter, self).__init__(name, values)

    # register the new hyper parameters
    hp.set_custom_parameter_class(MockDiscreteHyperParameter)

    params = get_hyperparameter_list()
    params.append(MockDiscreteHyperParameter('mock-param', ['x', 'y']))

    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    # serialization
    dataset.save_dataset()

    assert len(dataset) == 5
    assert os.path.exists(dataset.data_path)
    assert os.path.exists(dataset.parameter_path)

    # deserialization
    dataset.clear()
    assert len(dataset) == 0

    dataset.restore_dataset()

    assert len(dataset) == 5
    assert os.path.exists(dataset.data_path)
    assert os.path.exists(dataset.parameter_path)

    # deserialization from class
    path = os.path.join('shac', 'datasets')
    dataset2 = data.Dataset.load_from_directory(path)

    assert dataset2.parameters is not None
    assert len(dataset2.X) == 5
    assert len(dataset2.Y) == 5
    assert len(dataset2) == 5

    assert 'mock-param' in dataset2.parameters.name_map.values()
    assert dataset2.parameters.num_choices == 5

    dataset3 = data.Dataset.load_from_directory()

    assert dataset3.parameters is not None
    assert len(dataset3.X) == 5
    assert len(dataset3.Y) == 5

    assert 'mock-param' in dataset3.parameters.name_map.values()
    assert dataset3.parameters.num_choices == 5

    # serialization of empty get_dataset
    dataset = data.Dataset()

    with pytest.raises(FileNotFoundError):
        dataset.load_from_directory('null')

    with pytest.raises(ValueError):
        dataset.save_dataset()


@cleanup_dirs
@deterministic_test
def test_dataset_single_encoding_decoding():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    sample = (h.sample(), np.random.uniform())
    dataset.add_sample(*sample)

    encoded_x, encoded_y = dataset.encode_dataset()
    y_values = [0.]

    assert encoded_x.shape == (1, 4)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (1,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)

    decoded_x = dataset.decode_dataset(encoded_x)
    assert decoded_x.shape == (1, 4)


@cleanup_dirs
@deterministic_test
def test_dataset_single_multi_encoding_decoding():
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    sample = (h.sample(), np.random.uniform())
    dataset.add_sample(*sample)

    encoded_x, encoded_y = dataset.encode_dataset()
    y_values = [0.]

    assert encoded_x.shape == (1, 14)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (1,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)

    decoded_x = dataset.decode_dataset(encoded_x)
    assert decoded_x.shape == (1, 14)


@cleanup_dirs
@deterministic_test
def test_dataset_single_encoding_decoding_min():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    sample = (h.sample(), np.random.uniform())
    dataset.add_sample(*sample)

    encoded_x, encoded_y = dataset.encode_dataset(objective='min')
    y_values = [0.]

    assert encoded_x.shape == (1, 4)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (1,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)

    decoded_x = dataset.decode_dataset(encoded_x)
    assert decoded_x.shape == (1, 4)


@cleanup_dirs
@deterministic_test
def test_dataset_single_multi_encoding_decoding_min():
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    sample = (h.sample(), np.random.uniform())
    dataset.add_sample(*sample)

    encoded_x, encoded_y = dataset.encode_dataset(objective='min')
    y_values = [0.]

    assert encoded_x.shape == (1, 14)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (1,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)

    decoded_x = dataset.decode_dataset(encoded_x)
    assert decoded_x.shape == (1, 14)


@cleanup_dirs
@deterministic_test
def test_dataset_encoding_decoding():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    encoded_x, encoded_y = dataset.encode_dataset(objective='min')
    y_values = [0., 1., 0., 1., 0.]

    assert encoded_x.shape == (5, 4)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (5,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)

    decoded_x = dataset.decode_dataset(encoded_x)
    decoded_x2 = dataset.decode_dataset()
    assert decoded_x.shape == (5, 4)
    assert len(decoded_x) == len(decoded_x2)

    x, y = dataset.get_dataset()
    x_ = x[:, :3].astype('float')
    decoded_x_ = decoded_x[:, :3].astype('float')
    assert np.allclose(x_, decoded_x_, rtol=1e-3)

    samples2 = [(h.sample(), np.random.uniform()) for _ in range(5)]
    x, y = zip(*samples2)

    encoded_x, encoded_y = dataset.encode_dataset(x, y, objective='min')
    y_values = [0., 0., 1., 1., 0.]

    assert encoded_x.shape == (5, 4)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (5,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)


@cleanup_dirs
@deterministic_test
def test_dataset_multi_encoding_decoding():
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    encoded_x, encoded_y = dataset.encode_dataset(objective='min')
    y_values = [0., 0., 1., 0., 1.]

    assert encoded_x.shape == (5, 14)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (5,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)

    decoded_x = dataset.decode_dataset(encoded_x)
    decoded_x2 = dataset.decode_dataset()
    assert decoded_x.shape == (5, 14)
    assert len(decoded_x) == len(decoded_x2)

    x, y = dataset.get_dataset()
    x_ = x[:, :10].astype('float')
    decoded_x_ = decoded_x[:, :10].astype('float')
    assert np.allclose(x_, decoded_x_, rtol=1e-3)

    samples2 = [(h.sample(), np.random.uniform()) for _ in range(5)]
    x, y = zip(*samples2)

    encoded_x, encoded_y = dataset.encode_dataset(x, y, objective='min')
    y_values = [0., 1., 0., 1., 0.]

    assert encoded_x.shape == (5, 14)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (5,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)


@cleanup_dirs
@deterministic_test
def test_dataset_encoding_decoding_min():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    encoded_x, encoded_y = dataset.encode_dataset(objective='min')
    y_values = [0., 1., 0., 1., 0.]

    assert encoded_x.shape == (5, 4)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (5,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)

    decoded_x = dataset.decode_dataset(encoded_x)
    assert decoded_x.shape == (5, 4)

    x, y = dataset.get_dataset()
    x_ = x[:, :3].astype('float')
    decoded_x_ = decoded_x[:, :3].astype('float')
    assert np.allclose(x_, decoded_x_, rtol=1e-3)

    samples2 = [(h.sample(), np.random.uniform()) for _ in range(5)]
    x, y = zip(*samples2)

    encoded_x, encoded_y = dataset.encode_dataset(x, y, objective='min')
    y_values = [0., 0., 1., 1., 0.]

    assert encoded_x.shape == (5, 4)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (5,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)


@cleanup_dirs
@deterministic_test
def test_dataset_multi_encoding_decoding_min():
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    dataset = data.Dataset(h)

    samples = [(h.sample(), np.random.uniform()) for _ in range(5)]
    for sample in samples:
        dataset.add_sample(*sample)

    encoded_x, encoded_y = dataset.encode_dataset(objective='min')
    y_values = [0., 0., 1., 0., 1.]

    assert encoded_x.shape == (5, 14)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (5,)
    assert encoded_y.dtype == np.float64
    assert np.allclose(y_values, encoded_y, rtol=1e-3)

    decoded_x = dataset.decode_dataset(encoded_x)
    assert decoded_x.shape == (5, 14)

    x, y = dataset.get_dataset()
    x_ = x[:, :10].astype('float')
    decoded_x_ = decoded_x[:, :10].astype('float')
    assert np.allclose(x_, decoded_x_, rtol=1e-3)

    samples2 = [(h.sample(), np.random.uniform()) for _ in range(5)]
    x, y = zip(*samples2)

    encoded_x, encoded_y = dataset.encode_dataset(x, y, objective='min')
    y_values = [0., 1., 0., 1., 0.]

    assert encoded_x.shape == (5, 14)
    assert encoded_x.dtype == np.float64
    assert encoded_y.shape == (5,)
    assert encoded_y.dtype == np.float64
    print(encoded_y)
    assert np.allclose(y_values, encoded_y, rtol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
