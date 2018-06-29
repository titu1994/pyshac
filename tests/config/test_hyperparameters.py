import numpy as np
import six
import pytest

import pyshac.config.hyperparameters as hp


def deterministic_test(func):
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        np.random.seed(0)
        output = func(*args, **kwargs)
        np.random.seed(None)

        return output
    return wrapper


def get_hyperparameter_list():
    h1 = hp.DiscreteHyperParameter('h1', [0, 1, 2])
    h2 = hp.DiscreteHyperParameter('h2', [3, 4, 5, 6])
    h3 = hp.UniformContinuousHyperParameter('h3', 7, 10)
    h4 = hp.DiscreteHyperParameter('h4', ['v1', 'v2'])
    return [h1, h2, h3, h4]


def test_abstract():
    with pytest.raises(TypeError):
        hp.AbstractHyperParameter('h1', [0, 1])

    with pytest.raises(TypeError):
        hp.AbstractHyperParameter('h2', None)


def test_discrete():
    h1 = hp.DiscreteHyperParameter('h1', [0, 1])

    assert h1.name == 'h1'
    assert h1.num_choices == 2
    assert 0 in h1.id2param.values()
    assert h1.param2id[0] == 0
    assert repr(h1)


def test_discrete_no_values():
    with pytest.raises(ValueError):
        hp.DiscreteHyperParameter(None, [0, 1])

    with pytest.raises(ValueError):
        hp.DiscreteHyperParameter('h1', None)

    with pytest.raises(ValueError):
        hp.DiscreteHyperParameter('h2', [])


def test_discrete_sample():
    values = [1, 2, 3, 4, 5]

    h1 = hp.DiscreteHyperParameter('h1', values)
    sample = h1.sample()
    assert sample in values

    samples = np.array([h1.sample() for _ in range(100)])
    _, counts = np.unique(samples, return_counts=True)
    assert np.all(counts > 0)


@deterministic_test
def test_discrete_encode_decode():
    values = [10, 11, 12, 13, 14]

    h1 = hp.DiscreteHyperParameter('h1', values)
    sample = h1.sample()

    encoded = h1.encode(sample)
    assert encoded == 4

    decoded = h1.decode(encoded)
    assert decoded == values[encoded]


def test_discrete_serialization_deserialization():
    h1 = hp.DiscreteHyperParameter('h1', [0, 1])

    config = h1.get_config()
    assert 'name' in config
    assert 'values' in config

    values = config['values']
    assert len(values) == 2

    h2 = hp.DiscreteHyperParameter.load_from_config(config)
    config = h2.get_config()

    assert 'name' in config
    assert 'values' in config

    values = config['values']
    assert len(values) == 2


def test_abstract_continuous():
    h1 = hp.AbstractContinuousHyperParameter('h1', 0.0, 1.0)

    assert h1.name == 'h1'
    assert h1.num_choices == 0
    assert h1._val1 == 0.0
    assert h1._val2 == 1.0
    assert repr(h1)


def test_abstract_continuous_no_values():
    with pytest.raises(ValueError):
        hp.AbstractContinuousHyperParameter(None, 0, 1)

    with pytest.raises(ValueError):
        hp.AbstractContinuousHyperParameter('h1', None, None)


@deterministic_test
def test_abstract_continuous_sample():
    h1 = hp.AbstractContinuousHyperParameter('h1', 0.0, 1.0)

    with pytest.raises(NotImplementedError):
        sample = h1.sample()


@deterministic_test
def test_abstact_continuous_encode_decode():
    h1 = hp.AbstractContinuousHyperParameter('h1', 0.0, 1.0)

    encoded = h1.encode(0.5)
    assert encoded == 0.5

    decoded = h1.decode(encoded)
    assert decoded == 0.5


@deterministic_test
def test_abstract_continuous_log_space_encode_decode():
    h1 = hp.AbstractContinuousHyperParameter('h1', 0.0, 1.0, log_encode=True)

    encoded = h1.encode(0.5)
    assert encoded == -0.6931471805599453

    decoded = h1.decode(encoded)
    assert decoded == 0.5

    encoded = h1.encode(0.0)
    assert encoded == -np.inf

    decoded = h1.decode(encoded)
    assert decoded == 0.0


def test_continuous_uniform():
    h1 = hp.UniformContinuousHyperParameter('h1', 0.0, 1.0)

    assert h1.name == 'h1'
    assert h1.num_choices == 0
    assert h1.min_value == 0.0
    assert h1.max_value == 1.0
    assert repr(h1)


def test_continuous_uniform_no_values():
    with pytest.raises(ValueError):
        hp.UniformContinuousHyperParameter(None, 0, 1)

    with pytest.raises(ValueError):
        hp.UniformContinuousHyperParameter('h1', None, None)


@deterministic_test
def test_continuous_uniform_sample():
    h1 = hp.UniformContinuousHyperParameter('h1', 0.0, 1.0)
    sample = h1.sample()
    assert 0.0 <= sample and sample < 1.0


@deterministic_test
def test_continuous_uniform_encode_decode():
    h1 = hp.UniformContinuousHyperParameter('h1', 0.0, 1.0)
    sample = h1.sample()

    encoded = h1.encode(sample)
    assert encoded == sample

    decoded = h1.decode(encoded)
    assert decoded == sample


@deterministic_test
def test_continuous_uniform_encode_decode_log_space():
    h1 = hp.UniformContinuousHyperParameter('h1', 0.0, 1.0, log_encode=True)
    sample = h1.sample()

    encoded = h1.encode(sample)
    assert encoded == -0.5999965965916227

    decoded = h1.decode(encoded)
    assert decoded == sample


def test_uniform_serialization_deserialization():
    h1 = hp.UniformContinuousHyperParameter('h1', 0.0, 1.0, log_encode=True)

    config = h1.get_config()
    assert 'name' in config
    assert 'min_value' in config
    assert 'max_value' in config
    assert 'log_encode' in config

    min, max = config['min_value'], config['max_value']
    assert min == 0.0
    assert max == 1.0

    h2 = hp.UniformContinuousHyperParameter.load_from_config(config)
    config = h2.get_config()

    assert 'name' in config
    assert 'min_value' in config
    assert 'max_value' in config
    assert 'log_encode' in config

    min, max = config['min_value'], config['max_value']
    assert min == 0.0
    assert max == 1.0


def test_continuous_normal():
    h1 = hp.NormalContinuousHyperParameter('h1', 0.0, 1.0)

    assert h1.name == 'h1'
    assert h1.num_choices == 0
    assert h1.mean == 0.0
    assert h1.std == 1.0
    assert repr(h1)


def test_continuous_normal_no_values():
    with pytest.raises(ValueError):
        hp.NormalContinuousHyperParameter(None, 0, 1)

    with pytest.raises(ValueError):
        hp.NormalContinuousHyperParameter('h1', None, None)


@deterministic_test
def test_continuous_normal_sample():
    h1 = hp.NormalContinuousHyperParameter('h1', 0.0, 1.0)
    sample = [h1.sample() for _ in range(10000)]
    mean = np.mean(sample)
    std = np.std(sample)
    assert np.allclose(mean, 0.0, atol=0.025)
    assert np.allclose(std, 1.0, atol=0.025)


@deterministic_test
def test_continuous_normal_encode_decode():
    h1 = hp.NormalContinuousHyperParameter('h1', 0.0, 1.0)
    sample = h1.sample()

    encoded = h1.encode(sample)
    assert encoded == sample

    decoded = h1.decode(encoded)
    assert decoded == sample


def test_normal_serialization_deserialization():
    h1 = hp.NormalContinuousHyperParameter('h1', 0.0, 1.0)

    config = h1.get_config()
    assert 'name' in config
    assert 'mean' in config
    assert 'std' in config

    mean, std = config['mean'], config['std']
    assert mean == 0.0
    assert std == 1.0

    h2 = hp.NormalContinuousHyperParameter.load_from_config(config)
    config = h2.get_config()

    assert 'name' in config
    assert 'mean' in config
    assert 'std' in config

    mean, std = config['mean'], config['std']
    assert mean == 0.0
    assert std == 1.0


def test_list():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    assert h.name == 'parameter_list'
    assert h.num_choices == 4
    assert repr(h)

    list_names = h.get_parameter_names()

    for param in params:
        assert param.name in list_names
        assert h.param2id[param.name] is not None


def test_list_empty():
    h = hp.HyperParameterList()

    assert h.name == 'parameter_list'
    assert h.num_choices == 0
    assert len(h.id2param) == 0
    assert len(h.param2id) == 0
    assert len(h.name_map) == 0


def test_list_add():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList()

    for param in params:
        h.add_hyper_parameter(param)

    assert h.name == 'parameter_list'
    assert h.num_choices == 4

    for param in params:
        assert param.name in h.name_map.values()
        assert h.param2id[param.name] is not None

    # add a parameter whose name already exists in name map
    with pytest.raises(ValueError):
        h.add_hyper_parameter(params[0])

    # add a null parameter
    with pytest.raises(ValueError):
        h.add_hyper_parameter(None)


def test_list_remove():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    # remove by parameter class
    h.remove_hyper_parameter(params[0])

    assert h.num_choices == 3
    assert params[0].name not in h.name_map.values()
    assert params[0].name not in h.param2id

    for param in params[1:]:
        assert param.name in h.name_map.values()
        assert h.param2id[param.name] is not None

    # remove by string name
    h.remove_hyper_parameter('h2')

    assert h.num_choices == 2
    assert params[1].name not in h.name_map.values()
    assert params[1].name not in h.param2id
    assert params[2].name in h.name_map.values()
    assert h.param2id[params[2].name] is not None

    with pytest.raises(KeyError):
        h.remove_hyper_parameter('h5')

    with pytest.raises(ValueError):
        h.remove_hyper_parameter(None)


def test_list_sample():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    sample = h.sample()
    assert len(sample) == 4


@deterministic_test
def test_list_encoded_decoded():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    sample = h.sample()
    encoded = h.encode(sample)
    encoding = [0., 3., 9.1455681, 1.]
    assert np.allclose(encoded, encoding, rtol=1e-5)

    decoded = h.decode(encoded)
    sample_ = sample[:3]
    decoded_ = decoded[:3]
    assert np.allclose(decoded_, sample_, rtol=1e-5)


@deterministic_test
def test_list_encoded_decoded_numpy():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    sample = np.array(h.sample())
    encoded = h.encode(sample)
    encoding = [0., 3., 9.1455681, 1.]
    assert np.allclose(encoded, encoding, rtol=1e-5)

    decoded = np.array(h.decode(encoded))
    decoded_ = decoded[:3].astype('float')
    sample_ = sample[:3].astype('float')
    assert np.allclose(decoded_, sample_, rtol=1e-5)

    sample = np.array([h.sample()])
    with pytest.raises(ValueError):
        h.encode(sample)
        h.decode(sample)


def test_list_serialization_deserialization():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    config = h.get_config()
    assert len(config) == len(h.name_map)

    cnames_config = config.values()
    for cls_name, cls_cfg in cnames_config:
        cls = hp.get_parameter(cls_name)

        assert cls.load_from_config(cls_cfg)

    h = hp.HyperParameterList.load_from_config(config)
    assert len(config) == len(h.name_map)

    cnames_config = list(config.values())
    for cname_cfg in cnames_config:
        cls_name, cls_cfg = cname_cfg
        cls = hp.get_parameter(cls_name)

        assert cls.load_from_config(cls_cfg)


def test_set_custom_parameter():
    class TempClass(hp.DiscreteHyperParameter):

        def __init__(self, name, values):
            super(TempClass, self).__init__(name, values)

    hp.set_custom_parameter_class(TempClass)

    assert hp.get_parameter('TempClass')


if __name__ == '__main__':
    pytest.main([__file__])
