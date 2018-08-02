

# Check out: https://www.tensorflow.org/guide/custom_estimators

import tensorflow as tf

# Understanding dataset

a = tf.random_uniform([4, 10])

dataset1 = tf.data.Dataset.from_tensor_slices(a)

iterator = dataset1.make_initializable_iterator()
next_element = iterator.get_next()


for i in range(100):
  value = sess.run(next_element)
  print (value)


