import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

"""
Tensorflow 1.8

Since this is a simple optimization problem, lets try using TF Eager to
get a solution. Since there exist many possible solutions to this problem,
we are going to set the random seed to be able to replicate our results
"""

tf.enable_eager_execution()
tf.set_random_seed(1000)

# Create the variables to be optimized
x = tf.get_variable('x', initializer=tf.random_uniform([], -5.0, 5.0))
y = tf.get_variable('y', initializer=tf.random_uniform([], -0.0, 15.0))


# define the squared error loss function as before
def loss(x, y):
    y_predicted = 2 * x - y
    y_true = 4.0
    return tf.square(y_predicted - y_true)


print("Initial loss : ", loss(x, y).numpy())

# Get the gradients and variables to be optimized
grad_fn = tfe.implicit_gradients(loss)
grad_vars = grad_fn(x, y)

# prepare the optimizer. Since this is a very simple problem, we don't need
# many optimization steps
optimizer = tf.train.GradientDescentOptimizer(0.01)

for i in range(10):
    # update the variables and print the loss value
    optimizer.apply_gradients(grad_vars)
    print("[%d] Loss = %0.6f - (x = %0.5f, y = %0.5f)" % (i + 1, loss(x, y).numpy(), x.numpy(), y.numpy()))

print()
print("Final Loss = %0.6f - (x = %0.5f, y = %0.5f)" % (loss(x, y).numpy(), x.numpy(), y.numpy()))

"""
For convex optimization problems like these, TF is a far better fit
than SHAC, since SHAC is a sampling algorithm which searches the
solution space, and has to take many samples to reduce the search
space sufficiently to get the best values.

SHAC is better suited for problems where it is necessary to search a
sample space with better heuristics - such as architecture search
and hyper parameter search, since full search over these large search
spaces can be extremely expensive.
"""
