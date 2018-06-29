# Installation
----

Installation of this project is dependent on XGBoost, which has a slightly difficult installation process.

## Installing XGBoost

Since this library depends heavily on XGBoost, we recommend following the installation instructions posted there
for installing XGBoost on your system.

We use the Scikit-Learn Python wrappers for XGBoost, which do not support GPU execution at the moment, and the models
themselves are trained on exceedingly small amounts of data, therefore we do not require GPU execution of XGBoost.

- [XGBoost : Install Instructions](https://xgboost.readthedocs.io/en/latest/build.html)
- Or via pip : `pip install --upgrade xgboost`

!!!warning "Windows Installation"
    For installation of XGBoost on Windows, it is preferred to use the unofficial binaries
    provided here if you do not wish to build the project yourself :

    [XGBoost : Unofficial Windows Binaries](http://www.picnet.com.au/blogs/guido/2016/09/22/xgboost-windows-x64-binaries-for-download/)

## Installation of PySHAC

Once the XGBoost package is installed and verifier, we can simply clone this repository and run
`python setup.py install` to install this package.

```
git clone https://github.com/titu1994/pyshac.git
cd pyshac
python setup.py install
```

## Installation of External Libraries

When using the managed engines, it is required to separately install external libraries such as :

- [Tensorflow](https://www.tensorflow.org/install/)
- [PyTorch](https://pytorch.org/)
- [Keras](https://keras.io/#installation)