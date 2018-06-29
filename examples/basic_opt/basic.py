import numpy as np
import pyshac


# define the evaluation function
def squared_error_loss(id, parameters):
    x = parameters['x']
    y = parameters['y']
    y_sample = 2 * x - y

    # assume best values of x and y and 2 and 0 respectively
    y_true = 4.

    return np.square(y_sample - y_true)


if __name__ == '__main__':  # this is required for Windows ; not for Unix or Linux

    # define the parameters
    param_x = pyshac.UniformContinuousHyperParameter('x', -5.0, 5.0)
    param_y = pyshac.UniformContinuousHyperParameter('y', -2.0, 2.0)

    parameters = [param_x, param_y]

    # define the total budget as 100 evaluations
    total_budget = 100  # 100 evaluations at maximum

    # define the number of batches
    num_batches = 10  # 10 samples per batch

    # define the objective
    objective = 'min'  # minimize the squared loss

    shac = pyshac.SHAC(squared_error_loss, parameters, total_budget, num_batches, objective)

    # train the classifiers
    # `early stopping` default is False, and it is preferred not to use it when using `relax checks`
    # shac.fit(skip_cv_checks=True, early_stop=False, relax_checks=False)

    # uncomment this if classifiers are already trained and only needs to predict
    shac.restore_data()

    # sample more than one batch of hyper parameters
    parameter_samples = shac.predict(1)  # samples 10 batches, each of batch size 10 (100 samples in total)

    losses = [squared_error_loss(0, params) for params in parameter_samples]
    x_list = [param['x'] for param in parameter_samples]
    y_list = [param['y'] for param in parameter_samples]

    for i, (x, y) in enumerate(zip(x_list, y_list)):
        print("Sample %d : (%0.4f, %0.4f)" % (i + 1, x, y))

    print()
    print("Mean squared error of samples : ", np.mean(losses))

