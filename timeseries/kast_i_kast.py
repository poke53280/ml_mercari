
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt



tfd = tfp.distributions

# 1. Create hidden ground truth
features = tfd.Uniform(low = -1.0, high = 1.0).sample(int(100))
# labels = tfp.distributions.Bernoulli(logits=1.618 * features).sample()

labels = tfd.Normal(loc = 13 * features, scale = 1.) .sample()

# 2. Probabilistic programming: Specify model.
model = tfp.glm.Normal()

# 3. Processing
# Fit model given data.
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=features[:, tf.newaxis],
    response=tf.cast(labels, dtype=tf.float32),
    model=model)

# 4. Results
print (coeffs)



########################################################################


# Define a single scalar Normal distribution.
dist = tfd.Normal(loc=13., scale=3.)



gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[0.97, 0.03]),
    components_distribution=tfd.Normal(
      loc=[7., 24],       # One for each component.
      scale=[2.1, 2.5]))  # And same here.

gm.mean()
# ==> 0.4

gm.variance()
# ==> 1.018


# Plot PDF.
x = np.linspace(0., 55., int(1e4), dtype=np.float32)

plt.plot(x, gm.prob(x).numpy());

plt.show()


gm.sample(10)

model = tfp.glm.Normal()

### Create a mixture of two Bivariate Gaussians:

gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[0.3, 0.7]),
    components_distribution=tfd.MultivariateNormalDiag(
        loc=[[-1., 1],  # component 1
             [1, -1]],  # component 2
        scale_identity_multiplier=[.3, .6]))

gm.mean()
# ==> array([ 0.4, -0.4], dtype=float32)

gm.covariance()
# ==> array([[ 1.119, -0.84],
#            [-0.84,  1.119]], dtype=float32)

# Plot PDF contours.
def meshgrid(x, y=x):
  [gx, gy] = np.meshgrid(x, y, indexing='ij')
  gx, gy = np.float32(gx), np.float32(gy)
  grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
  return grid.T.reshape(x.size, y.size, 2)
grid = meshgrid(np.linspace(-2, 2, 100, dtype=np.float32))
plt.contour(grid[..., 0], grid[..., 1], gm.prob(grid).eval());