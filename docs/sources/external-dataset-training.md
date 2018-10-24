# External Dataset Training
-----

In many cases, one may posses results from other search processes such as bayesian optimization, which is used for
hyper parameter tuning or even neural architecture search.

However, they come with their own set of limitations, and can often be very costly for searching over
very large search spaces.

Therefore, we can utilize the results derived from such external processes and utilize them to train
SHAC engines quickly, which can then be used for speedy inference of optimal parameters in large search
spaces.

## Formatting External Datasets
-----

PySHAC uses a standard CSV file to define the contents of the dataset that it generates, which makes it highly
convenient to create or transform external datasets into a format that can be readily used by the engine.

!!!note "Standard format of datasets"
    Each dataset csv file must contain an integer id column named "id"
    as its 1st column, followed by several columns describing the values
    taken by the hyper parameters, and the final column must be for
    the the objective criterion, and *must* be named "scores".
    The csv file *must* contain a header, following the above format.

Example:

    id,hp1,hp2,scores
    0,1,h1,1.0
    1,1,h2,0.2
    2,0,h1,0.0
    3,0,h3,0.5
    ...

## Training with an External Dataset
-----

There are a few requirements for the engine when loading external datasets.

1) The parameter list provided to the engine must match the parameter list of the dataset. It is not
possible to provide an empty HyperParameter list when training via an external dataset.

2) The number of samples in the dataset must be greater than or equal to the total budget. If it is
greater than the total budget, the additional samples will not be used for training.

Once these requirements are met, it is as simple as calling a single function of the engine to
train a model using external data.

```python
shac.fit_dataset('path/to/dataset.csv', presort=True)
```

The `presort` option will first sort the dataset automatically in ascending or descending order with respect to
the objective function optimization flag provided to the engine, and this can often improve the performance
of the final cascade of classifiers of the SHAC engine.

After training, which should be extremely fast for even large datasets, we can predict new samples using
`predict` as always without any modifications.

## Example

An example script showing the usage of external dataset training is provided in the [`examples/basic_dataset` folder](https://github.com/titu1994/pyshac/tree/master/examples/basic_dataset)