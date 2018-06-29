import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

"""
Tensorflow 1.8

Since this is a search space problem, lets try using TF Eager to
get a solution. Likely, gradient optimization algorithms will diverge
eventually.
"""

tf.enable_eager_execution()
tf.set_random_seed(1000)

# Create the variables to be optimized
x = tf.get_variable('x', initializer=tf.random_uniform([], -5.0, 10.0))
y = tf.get_variable('y', initializer=tf.random_uniform([], -2.0, 2.0))


def evaluation_branin(x, y):
    """ Code ported from https://www.sfu.ca/~ssurjano/Code/braninm.html
    Global Minimum = -0.397887
    """

    a = 1.0
    b = 5.1 / (4 * (np.pi ** 2))
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    term1 = a * ((y - b * (x ** 2) + c * x - r) ** 2)
    term2 = s * (1.0 - t) * tf.cos(x)

    out = term1 + term2 + s
    return out


def check_branin_impl():
    # Optimal parameter 1
    x = [-np.pi, 12.275]
    params = [tf.get_variable('%d' % i, initializer=xi)
              for i, xi in enumerate(x)]

    loss = evaluation_branin(*params).numpy()
    assert np.allclose(loss, 0.397887)

    # Optimal parameter 2
    x = [np.pi, 2.275]
    params = [tf.get_variable('%d' % i, initializer=xi)
              for i, xi in enumerate(x)]

    loss = evaluation_branin(*params).numpy()
    assert np.allclose(loss, 0.397887)

    # Optimal parameter 3
    x = [9.42478, 2.475]
    params = [tf.get_variable('%d' % i, initializer=xi)
              for i, xi in enumerate(x)]

    loss = evaluation_branin(*params).numpy()
    assert np.allclose(loss, 0.397887)


check_branin_impl()

print("Initial loss : ", evaluation_branin(x, y).numpy())

# Get the gradients and variables to be optimized
grad_fn = tfe.implicit_gradients(evaluation_branin)
grad_vars = grad_fn(x, y)

# {repare the optimizer. Since this is a very simple problem, we don't need
# many optimization steps
optimizer = tf.train.AdamOptimizer(0.01)

for i in range(200):
    # update the variables and print the loss value
    optimizer.apply_gradients(grad_vars)
    print("[%d] Loss = %0.6f - (x = %0.5f, y = %0.5f)" % (i + 1, evaluation_branin(x, y).numpy(), x.numpy(), y.numpy()))

print()
print("Final Loss = %0.6f - (x = %0.5f, y = %0.5f)" % (evaluation_branin(x, y).numpy(), x.numpy(), y.numpy()))

"""
As can be seen when this script is run, since this problem is not a simple
optimization objective, gradient descent can jump around and move past the
optimal region quite easily.

In this case, AdamOptimizer is able to reach 0.398831 at the 163rd time step,
after which it begins to diverge. Using a learning rate lower than 0.01,
decaying it linearly or exponentially, or using SGD instead does not help
that much, and they all diverge eventually.

"""
