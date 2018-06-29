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

variables = [tf.get_variable('x%d' % i, initializer=tf.random_uniform([], 0.0, 1.0))
             for i in range(6)]


def evaluation_hartmann6(params):
    """ Code ported from https://www.sfu.ca/~ssurjano/Code/hart6scm.html
    Global Minimum = -3.32237
    """
    alpha = np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float32)

    A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], dtype=np.float32)

    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]], dtype=np.float32)

    xx = tf.reshape(params, (6, 1))

    inner = tf.reduce_sum(A * ((tf.transpose(xx) - P) ** 2), axis=-1)
    inner = tf.exp(-inner)
    outer = tf.reduce_sum(alpha * inner)

    return -outer


def check_hartmann6_impl():
    # Optimal _parameters
    x = [0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573]
    params = [tf.get_variable('x%d' % i, initializer=xi)
              for i, xi in enumerate(x)]

    loss = evaluation_hartmann6(params).numpy()

    assert np.allclose(loss, -3.32237)


check_hartmann6_impl()

print("Initial loss : ", evaluation_hartmann6(variables).numpy())


# Get the gradients and variables to be optimized
grad_fn = tfe.implicit_gradients(evaluation_hartmann6)
grad_vars = grad_fn(variables)

# {repare the optimizer. Since this is a very simple problem, we don't need
# many optimization steps
optimizer = tf.train.AdamOptimizer(0.001)

for i in range(500):
    # update the variables and print the loss value
    optimizer.apply_gradients(grad_vars)
    print("[%d] Loss = %0.6f" % (i + 1, evaluation_hartmann6(variables).numpy()))

print()
print("Final Loss = %0.6f" % (evaluation_hartmann6(variables).numpy()))


"""
As can be seen when this script is run, since this problem is not a simple
optimization objective, gradient descent based optimization is unable to reach
the true global minima, at least not easily.

In this case, AdamOptimizer is able to reach -1.315204 at the 238/th time step,
after which it begins to diverge. Using a learning rate lower than 0.1,
decaying it linearly or exponentially, or using SGD instead does not help
that much, and they all diverge eventually.

"""

