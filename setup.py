from setuptools import setup, find_packages
import re
import os


long_description = (
    'Python library which implements Succesive Halving and Classification '
    'for Parallel Architecture and Hyper Parameter Search from the paper '
    '[Parallel Architecture and Hyperparameter Search via Successive Halving '
    'and Classification](https://arxiv.org/abs/1805.10255).'
)


def get_version(package):
    """Return package version as listed in `__version__` in `init.py`."""
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


setup(
    name='pyshac',
    version=get_version("pyshac"),
    packages=find_packages(),
    url='',
    download_url='',
    license='MIT',
    author='Somshubra Majumdar',
    author_email='titu1994@gmail.com',
    description='Python library which implements Succesive Halving and Classification algorithm',
    long_description=long_description,
    install_requires=['numpy>=1.14.2',
                      'scikit-learn>=0.19.1',
                      'pandas>=0.19.2',
                      'joblib>=0.11',
                      'six>=1.11.0',
                      'xgboost>=0.72'],
    zip_safe=False,
)
