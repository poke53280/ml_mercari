
#
# https://machinelearningmastery.com/what-is-bayesian-optimization/
#
#
# Global optimization:
#
#   Given objective function.
#   Given cost function.
#   Given initial input.
#   
#   Measure cost of objective function.
#   Adjust input such that cost is minimal.
#
#
# Bayesian Optimization provides a principled technique based on Bayes Theorem 
# to direct a search of a global optimization problem that is efficient and effective.
#
#   Build a probabilistic model of the objective function. This is called the surrogate function.
#   Effeciently search the surrogate function with an acquisition function.
#   Choose candidate samples.
#   Evaluate the candidate samples on the real objective function.
#
#


#   Samples are drawn from the domain and evaluated by the objective function to give a score or cost.

# DEFINITIONS:

# SAMPLE. One example from the domain, represented as a vector.
# SEARCH SPACE. Extent of the domain from which samples can be drawn.
# OBJECTIVE FUNCTION. Function that takes a sample and returns a cost.
# COST. Numeric score for a sample calculated via the objective function.


# THE SAMPLE
# Samples are comprised of one or more variables generally easy to devise or create
# One sample is often defined as a vector of variables with a predefined range in an n-dimensional space.
# This space must be sampled and explored in order to find the specific combination of variable values that result in the best cost.

# THE COST
# The cost often has units that are specific to a given domain.
# Optimization is often described in terms of minimizing cost, as a maximization problem can easily be transformed into a minimization problem
# by inverting the calculated cost. Together, the minimum and maximum of a function are referred to as the extreme of the function (or the plural extrema).

# THE OBJECTIVE FUNCTION
# The objective function is often easy to specify but can be computationally challenging to calculate or
# result in a noisy calculation of cost over time. The form of the objective function is unknown and
# is often highly nonlinear, and highly multi-dimensional defined by the number of input variables.
# The function is also probably non-convex. This means that local extrema may or may not be the global extrema 
# (e.g. could be misleading and result in premature convergence), hence the name of the task as global rather than local optimization.

# Although little is known about the objective function, (it is known whether the minimum or the maximum cost from the function is sought),
# and as such, it is often referred to as a black box function and the search process as black box optimization.
#
# Further, the objective function is sometimes called an oracle given the ability to only give answers.


# Optimization in machine learning:
# 
# Algorithm Training. Optimization of model parameters.
# Algorithm Tuning. Optimization of model hyperparameters.
# Predictive Modeling. Optimization of data, data preparation, and algorithm selection.

#
# Many methods exist for function optimization, such as randomly sampling the variable search space,
# called random search, or systematically evaluating samples in a grid across the search space, called grid search.
#

# More principled methods are able to learn from sampling the space so that future samples are directed
# toward the parts of the search space that are most likely to contain the extrema.


# A directed approach to global optimization that uses probability is called Bayesian Optimization.


# What Is Bayesian Optimization

# Bayesian Optimization is an approach that uses Bayes Theorem to direct the search in
# order to find the minimum or maximum of an objective function.


# posterior = likelihood * prior

from matplotlib import pyplot as plt


# objective function
def objective(x, noise=0.1):
	noise = np.random.normal(loc=0, scale=noise)
	return (x**2 * np.sin(5 * np.pi * x)**6.0) + noise


objective(3)

# grid-based sample of the domain [0,1]
X = np.arange(0, 1, 0.01)

# sample the domain without noise
y = [objective(x, 0) for x in X]

# sample the domain with noise
ynoise = [objective(x) for x in X]

# find best result
ix = np.argmax(y)
print('Optima: x=%.3f, y=%.3f' % (X[ix], y[ix]))


# plot the points with noise
plt.scatter(X, ynoise)
# plot the points without noise
plt.plot(X, y)
# show the plot
plt.show()