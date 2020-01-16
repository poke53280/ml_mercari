

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


features = tfp.distributions.Normal(loc=0., scale=1.).sample(10)
np.array(features).max()


def bernoulli_analysis(p, n):

    anBernoulli = np.array(tfp.distributions.Bernoulli(probs=[p]).sample(n))

    anBernoulli = np.ravel(anBernoulli)

    nZero   = (anBernoulli == 0).sum()
    nOne    = (anBernoulli == 1).sum()
    nTotal  = nZero + nOne

    print(f"{nTotal}: #0: {nZero} #1: {nOne}")



bernoulli_analysis(0.5, 300)

p = np.exp(1) / (np.exp(1) + 1)

np.log(0.5/ (1 - 0.5))


np.log( 0.999 / (1 - 0.999))


features
labels = tfp.distributions.Bernoulli(logits= np.array([7,7,0,0,0])).sample()


# Pretend to load synthetic data set.
features = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
labels = tfp.distributions.Bernoulli(logits=1.618 * features).sample()

# Specify model.
model = tfp.glm.Bernoulli()

# Fit model given data.
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=features[:, tf.newaxis],
    response=tf.cast(labels, dtype=tf.float32),
    model=model)
# ==> coeffs is approximately [1.618] (We're golden!)