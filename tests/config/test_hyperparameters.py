import numpy as np
import pytest
import six

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


def get_multi_parameter_list():
    h1 = hp.MultiDiscreteHyperParameter('h1', [0, 1, 2], sample_count=2)
    h2 = hp.MultiDiscreteHyperParameter('h2', [3, 4, 5, 6], sample_count=3)
    h3 = hp.MultiUniformContinuousHyperParameter('h3', 7, 10, sample_count=5)
    h4 = hp.MultiDiscreteHyperParameter('h4', ['v1', 'v2'], sample_count=4)
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


def test_discrete_encode_decode():
    values = [10, 11, 12, 13, 14]

    h1 = hp.DiscreteHyperParameter('h1', values, seed=0)
    sample = h1.sample()

    encoded = h1.encode(sample)
    assert encoded == 4

    decoded = h1.decode(encoded)
    assert decoded == values[encoded]

    # Test for None input
    values = [None, 1, 2, 3]
    h2 = hp.DiscreteHyperParameter('h1', values, seed=0)
    sample = None

    encoded = h2.encode(sample)
    assert encoded == 0

    decoded = h2.decode(encoded)
    assert decoded == values[encoded]


def test_discrete_serialization_deserialization():
    h1 = hp.DiscreteHyperParameter('h1', [0, 1, None])

    config = h1.get_config()
    assert 'name' in config
    assert 'values' in config

    values = config['values']
    assert len(values) == 3

    h2 = hp.DiscreteHyperParameter.load_from_config(config)
    config = h2.get_config()

    assert 'name' in config
    assert 'values' in config

    values = config['values']
    assert len(values) == 3


def test_multi_discrete():
    h1 = hp.MultiDiscreteHyperParameter('h1', [0, 1], sample_count=5)

    assert h1.name == 'h1'
    assert h1.num_choices == 2
    assert 0 in h1.id2param.values()
    assert h1.param2id[0] == 0
    assert repr(h1)
    assert h1.sample_count > 0


def test_multi_discrete_no_values():
    with pytest.raises(ValueError):
        hp.MultiDiscreteHyperParameter(None, [0, 1])

    with pytest.raises(ValueError):
        hp.MultiDiscreteHyperParameter('h1', None)

    with pytest.raises(ValueError):
        hp.MultiDiscreteHyperParameter('h2', [])


def test_multi_discrete_sample():
    values = [1, 2, 3, 4, 5]

    h1 = hp.MultiDiscreteHyperParameter('h1', values, sample_count=10)
    sample = h1.sample()
    assert sample[0] in values
    assert len(sample) == 10

    samples = np.array([h1.sample() for _ in range(10)])
    _, counts = np.unique(samples, return_counts=True)
    assert np.all(counts > 0)


def test_multi_discrete_encode_decode():
    values = [10, 11, 12, 13, 14]

    h1 = hp.MultiDiscreteHyperParameter('h1', values, sample_count=5, seed=0)
    sample = h1.sample()

    encoded = h1.encode(sample)
    assert encoded == [4, 0, 3, 3, 3]

    decoded = h1.decode(encoded)
    for i in range(len(decoded)):
        assert decoded[i] == values[encoded[i]]

    # Test for None input
    values = [None, 1, 2, 3]
    h2 = hp.MultiDiscreteHyperParameter('h1', values, sample_count=10, seed=0)
    sample = h2.sample()

    encoded = h2.encode(sample)
    assert encoded == [0, 3, 1, 0, 3, 3, 3, 3, 1, 3]

    decoded = h2.decode(encoded)
    for i in range(len(decoded)):
        assert decoded[i] == values[encoded[i]]


def test_multi_discrete_serialization_deserialization():
    h1 = hp.MultiDiscreteHyperParameter('h1', [0, 1, None], sample_count=5)

    config = h1.get_config()
    assert 'name' in config
    assert 'values' in config
    assert 'sample_count' in config

    values = config['values']
    assert len(values) == 3
    assert config['sample_count'] == 5

    h2 = hp.MultiDiscreteHyperParameter.load_from_config(config)
    config = h2.get_config()

    assert 'name' in config
    assert 'values' in config
    assert 'sample_count' in config

    values = config['values']
    assert len(values) == 3
    assert config['sample_count'] == 5


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


def test_abstract_multi_continuous():
    h1 = hp.AbstractMultiContinuousHyperParameter('h1', 0.0, 1.0, sample_count=5)

    assert h1.name == 'h1'
    assert h1.num_choices == 0
    assert h1._val1 == 0.0
    assert h1._val2 == 1.0
    assert repr(h1)
    assert h1.sample_count == 5


def test_abstract_multi_continuous_no_values():
    with pytest.raises(ValueError):
        hp.AbstractMultiContinuousHyperParameter(None, 0, 1)

    with pytest.raises(ValueError):
        hp.AbstractMultiContinuousHyperParameter('h1', None, None)


@deterministic_test
def test_abstract_multi_continuous_sample():
    h1 = hp.AbstractMultiContinuousHyperParameter('h1', 0.0, 1.0, sample_count=5)

    with pytest.raises(NotImplementedError):
        sample = h1.sample()


@deterministic_test
def test_abstact_multi_continuous_encode_decode():
    h1 = hp.AbstractMultiContinuousHyperParameter('h1', 0.0, 1.0, sample_count=5)

    encoded = h1.encode([0.5, 0.2, 0.1, 0.3, 0.7])
    assert encoded == [0.5, 0.2, 0.1, 0.3, 0.7]

    decoded = h1.decode(encoded)
    assert decoded == [0.5, 0.2, 0.1, 0.3, 0.7]


@deterministic_test
def test_abstract_multi_continuous_log_space_encode_decode():
    h1 = hp.AbstractMultiContinuousHyperParameter('h1', 0.0, 1.0, log_encode=True,
                                                  sample_count=5)

    encoded = h1.encode([0.5, 0.2, 0.1, 0.3, 0.7])
    assert np.allclose(encoded, [-0.6931471805599453, -1.6094379124341003, -2.3025850929940455,
                       -1.2039728043259361, -0.35667494393873245], atol=1e-2)

    decoded = h1.decode(encoded)
    assert np.allclose(decoded, [0.5, 0.2, 0.1, 0.3, 0.7], atol=1e-3)

    encoded = h1.encode([0.0])
    assert encoded == [-np.inf]

    decoded = h1.decode(encoded)
    assert decoded == [0.0]


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


def test_continuous_multi_uniform():
    h1 = hp.MultiUniformContinuousHyperParameter('h1', 0.0, 1.0, sample_count=5)

    assert h1.name == 'h1'
    assert h1.num_choices == 0
    assert h1.min_value == 0.0
    assert h1.max_value == 1.0
    assert repr(h1)
    assert h1.sample_count == 5


def test_continuous_multi_uniform_no_values():
    with pytest.raises(ValueError):
        hp.MultiUniformContinuousHyperParameter(None, 0, 1)

    with pytest.raises(ValueError):
        hp.MultiUniformContinuousHyperParameter('h1', None, None)


@deterministic_test
def test_continuous_multi_uniform_sample():
    h1 = hp.MultiUniformContinuousHyperParameter('h1', 0.0, 1.0, sample_count=5)
    samples = h1.sample()
    assert all([0.0 <= sample and sample < 1.0 for sample in samples])


@deterministic_test
def test_continuous_multi_uniform_encode_decode():
    h1 = hp.MultiUniformContinuousHyperParameter('h1', 0.0, 1.0, sample_count=5)
    sample = h1.sample()

    encoded = h1.encode(sample)
    assert encoded == sample

    decoded = h1.decode(encoded)
    assert decoded == sample


def test_continuous_uniform_encode_decode_log_space():
    h1 = hp.MultiUniformContinuousHyperParameter('h1', 0.0, 1.0,
                                                 log_encode=True,
                                                 sample_count=3,
                                                 seed=0)
    sample = h1.sample()

    encoded = h1.encode(sample)
    assert encoded == [-0.5999965965916227, -0.3352079232808751,
                       -0.5062305704264942]

    decoded = h1.decode(encoded)
    assert decoded == sample


def test_multi_uniform_serialization_deserialization():
    h1 = hp.MultiUniformContinuousHyperParameter('h1', 0.0, 1.0, log_encode=True,
                                                 sample_count=5)

    config = h1.get_config()
    assert 'name' in config
    assert 'min_value' in config
    assert 'max_value' in config
    assert 'log_encode' in config
    assert 'sample_count' in config

    min, max = config['min_value'], config['max_value']
    assert min == 0.0
    assert max == 1.0
    assert config['sample_count'] == 5

    h2 = hp.MultiUniformContinuousHyperParameter.load_from_config(config)
    config = h2.get_config()

    assert 'name' in config
    assert 'min_value' in config
    assert 'max_value' in config
    assert 'log_encode' in config
    assert 'sample_count' in config

    min, max = config['min_value'], config['max_value']
    assert min == 0.0
    assert max == 1.0
    assert config['sample_count'] == 5


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


def test_continuous_multi_normal():
    h1 = hp.MultiNormalContinuousHyperParameter('h1', 0.0, 1.0, sample_count=5)

    assert h1.name == 'h1'
    assert h1.num_choices == 0
    assert h1.mean == 0.0
    assert h1.std == 1.0
    assert repr(h1)
    assert h1.sample_count == 5


def test_continuous_multi_normal_no_values():
    with pytest.raises(ValueError):
        hp.MultiNormalContinuousHyperParameter(None, 0, 1)

    with pytest.raises(ValueError):
        hp.MultiNormalContinuousHyperParameter('h1', None, None)


@deterministic_test
def test_continuous_normal_sample():
    h1 = hp.MultiNormalContinuousHyperParameter('h1', 0.0, 1.0, sample_count=1000)
    sample = [h1.sample() for _ in range(10)]
    mean = np.mean(sample)
    std = np.std(sample)
    assert np.allclose(mean, 0.0, atol=0.025)
    assert np.allclose(std, 1.0, atol=0.025)


@deterministic_test
def test_continuous_multi_normal_encode_decode():
    h1 = hp.MultiNormalContinuousHyperParameter('h1', 0.0, 1.0, sample_count=3)
    sample = h1.sample()

    encoded = h1.encode(sample)
    assert encoded == sample

    decoded = h1.decode(encoded)
    assert decoded == sample


def test_multi_normal_serialization_deserialization():
    h1 = hp.MultiNormalContinuousHyperParameter('h1', 0.0, 1.0, sample_count=5)

    config = h1.get_config()
    assert 'name' in config
    assert 'mean' in config
    assert 'std' in config
    assert 'sample_count' in config

    mean, std = config['mean'], config['std']
    assert mean == 0.0
    assert std == 1.0
    assert config['sample_count'] == 5

    h2 = hp.MultiNormalContinuousHyperParameter.load_from_config(config)
    config = h2.get_config()

    assert 'name' in config
    assert 'mean' in config
    assert 'std' in config
    assert 'sample_count' in config

    mean, std = config['mean'], config['std']
    assert mean == 0.0
    assert std == 1.0
    assert config['sample_count'] == 5


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

    # Multi Parameter tests
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    assert h.name == 'parameter_list'
    assert h.num_choices == 4
    assert repr(h)

    list_names = h.get_parameter_names()

    for param in params:
        assert h.param2id[param.name] is not None

        for i in range(param.sample_count):
            param_name = (param.name + '_%d' % (i + 1))
            assert param_name in list_names


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

    # Multi parameter tests
    params = get_multi_parameter_list()
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

    # Multi parameter tests
    params = get_multi_parameter_list()
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

    # Multi parameter tests
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    sample = h.sample()
    assert len(sample) == 14


def test_list_sample_seeded():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params, seed=0)

    sample = h.sample()
    assert len(sample) == 4
    assert sample == [0, 4, 8.307984706426012, 'v1']

    # Multi parameter tests
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params, seed=0)

    sample = h.sample()
    assert len(sample) == 14
    assert sample == [0, 1, 4, 6, 3,
                      8.307984706426012,
                      7.077778695483674,
                      8.648987433636128,
                      8.30596717785483,
                      8.261103406262468,
                      'v1', 'v1', 'v2', 'v2']


def test_list_encoded_decoded():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params, seed=0)

    sample = h.sample()
    encoded = h.encode(sample)
    encoding = [0., 1., 8.30798471, 0.]
    assert np.allclose(encoded, encoding, rtol=1e-5)

    decoded = h.decode(encoded)
    sample_ = sample[:3]
    decoded_ = decoded[:3]
    assert np.allclose(decoded_, sample_, rtol=1e-5)

    # Multi parameter tests
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params, seed=0)

    sample = h.sample()
    encoded = h.encode(sample)
    encoding = [0., 1., 1., 3., 0.,
                8.30798471, 7.0777787,
                8.64898743, 8.30596718, 8.26110341,
                0., 0., 1., 1.]
    print(encoded)
    assert np.allclose(encoded, encoding, rtol=1e-5)

    decoded = h.decode(encoded)
    sample_ = sample[:10]
    decoded_ = decoded[:10]
    assert np.allclose(decoded_, sample_, rtol=1e-5)


def test_list_encoded_decoded_numpy():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params, seed=0)

    sample = np.array(h.sample())
    encoded = h.encode(sample)
    encoding = [0., 1., 8.30798471, 0]
    assert np.allclose(encoded, encoding, rtol=1e-5)

    decoded = np.array(h.decode(encoded))
    decoded_ = decoded[:3].astype('float')
    sample_ = sample[:3].astype('float')
    assert np.allclose(decoded_, sample_, rtol=1e-5)

    sample = np.array([h.sample()])
    with pytest.raises(ValueError):
        h.encode(sample)
        h.decode(sample)

    # Multi parameter tests
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params, seed=0)

    sample = np.array(h.sample())
    encoded = h.encode(sample)
    encoding = [0., 1., 1., 3., 0.,
                8.30798471, 7.0777787,
                8.64898743, 8.30596718, 8.26110341,
                0., 0., 1., 1.]
    assert np.allclose(encoded, encoding, rtol=1e-5)

    decoded = np.array(h.decode(encoded))
    decoded_ = decoded[:10].astype('float')
    sample_ = sample[:10].astype('float')
    assert np.allclose(decoded_, sample_, rtol=1e-5)

    sample = np.array([h.sample()])
    with pytest.raises(ValueError):
        h.encode(sample)
        h.decode(sample)


def test_list_serialization_deserialization():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params, seed=0)

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

    # Multi parameter tests
    params = get_multi_parameter_list()
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


def test_list_cast():
    params = get_hyperparameter_list()
    h = hp.HyperParameterList(params)

    samples = h.sample()
    casted_samples = h._cast(samples)

    dtypes = [type(s) for s in samples]
    casted_dtypes = [type(s) for s in casted_samples]

    for s, cs, d, cd in zip(samples, casted_samples, dtypes, casted_dtypes):
        assert d == cd
        assert s == cs

    # Multi Parameter tests
    params = get_multi_parameter_list()
    h = hp.HyperParameterList(params)

    dtypes = [type(s) for s in samples]
    casted_dtypes = [type(s) for s in casted_samples]

    for s, cs, d, cd in zip(samples, casted_samples, dtypes, casted_dtypes):
        assert d == cd
        assert s == cs


def test_set_custom_parameter():
    class TempClass(hp.DiscreteHyperParameter):

        def __init__(self, name, values):
            super(TempClass, self).__init__(name, values)

    hp.set_custom_parameter_class(TempClass)

    assert hp.get_parameter('TempClass')

    # Multi parameter tests
    class TempClass2(hp.MultiDiscreteHyperParameter):

        def __init__(self, name, values, sample_count=5):
            super(TempClass2, self).__init__(name, values, sample_count)

    hp.set_custom_parameter_class(TempClass2)

    assert hp.get_parameter('TempClass2')


if __name__ == '__main__':
    pytest.main([__file__])
