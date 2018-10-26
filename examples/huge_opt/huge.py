import numpy as np
import pyshac
from pyshac.config.callbacks import CSVLogger
from pyshac.utils.vis_utils import plot_dataset

"""
Similar to the `basic` examples, but on an extremely large search space.
Designed to test the speed of convergence on extremely large search spaces.
"""


# define the evaluation function
def squared_error_loss(id, parameters):
    params = np.array(list(parameters.values()))
    x_params = params[:50]
    y_params = params[50:]

    x = np.mean(x_params)
    y = np.mean(y_params)

    y_sample = 2 * x - y

    # assume best values of x and y and 2 and 0 respectively
    y_true = 4.

    return np.square(y_sample - y_true)


# define the parameters
param_x = [pyshac.UniformHP('x%d' % i, -5.0, 5.0) for i in range(50)]
param_y = [pyshac.UniformHP('y%d' % i, -2.0, 2.0) for i in range(100)]
param_x.extend(param_y)

parameters = param_x

# define the total budget as 100 evaluations
total_budget = 1100  # 200 evaluations at maximum

# define the number of batches
num_batches = 11  # 10 samples per batch

# define the objective
objective = 'min'  # minimize the squared loss

shac = pyshac.SHAC(parameters, total_budget, num_batches, objective)

# train the classifiers
# `early stopping` default is False, and it is preferred not to use it when using `relax checks`
# shac.fit(squared_error_loss, skip_cv_checks=True, early_stop=False, relax_checks=False,
#          callbacks=[CSVLogger('logs.csv')])

# uncomment this if classifiers are already trained and only needs to predict
shac.restore_data()

# sample more than one batch of hyper parameters
parameter_samples = shac.predict(100)  # samples 100 hyper parameters

losses = [squared_error_loss(0, params) for params in parameter_samples]

print()
print("Mean squared error of samples : ", np.mean(losses))

plot_dataset(shac.dataset, eval_label='MSE')
