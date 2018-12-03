
import numpy as np
import pandas as pd
import gc
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Flatten

from keras.layers import Subtract
from keras.layers import LSTM
from keras.layers import TimeDistributed


from keras.models import Model

from keras.constraints import unitnorm

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=200)
np.core.arrayprint._line_width = 480


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE



with tf.Session() as sess:

    sess.run(iterator.initializer, feed_dict={encoded_placeholder: anData, raw_placeholder: anData_d})
    
    next_element = iterator.get_next()

    data_record = sess.run(next_element)
    print(data_record)

"""c"""


def input_fn():
  
    anData_d = np.load(DATA_DIR + "anData_d_all_2.npy")
    anData = np.load(DATA_DIR + "anData_all_2.npy")

    anData_d.shape[0] == anData.shape[0]


    encoded_placeholder = tf.placeholder(anData.dtype, anData.shape, "encoded_placeholderanData")
    raw_placeholder = tf.placeholder(anData_d.dtype, anData_d.shape, "raw_placeholderanData_d")

    ds = tf.data.FixedLengthRecordDataset.from_tensor_slices((encoded_placeholder, raw_placeholder))

    ds = dataset.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=50000)

    ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(8))

    return ds
"""c"""


def create_model_tf():


    vocab_size = 700
    emb_dim = 6

     # This is where training samples and labels are fed to the graph.

    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.

    train_data_node = tf.placeholder(tf.uint16, shape=(32, num_sequence_length * 6))

    train_data_out_node = tf.placeholder(tf.float32, shape=(32, num_sequence_length * 6))

    word_embeddings = tf.get_variable("word_embeddings", [vocab_size, emb_dim])
    
    # embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)

    x = l.Dense(64) (x)
    x = l.Flatten() (x)
    x = l.Dense(num_sequence_length * 6) (x)
  


def create_model_keras():
    l = tf.keras.layers

    num_sequence_length = 200

    input = tf.keras.Input(shape=(num_sequence_length * 6,), name="Encoder_input")

    vocab_size = 700
    emb_dim = 6

    x = l.Embedding(output_dim=emb_dim, input_dim=vocab_size, name="Embedding", trainable=True) (input)

    x = l.Dense(64) (x)
    x = l.Flatten() (x)
    x = l.Dense(num_sequence_length * 6) (x)

    model = tf.keras.Model(inputs=input, outputs=x)

    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metric='mse', run_opts = run_opts)

    est_model = tf.keras.estimator.model_to_estimator(keras_model=model)

    model.input_names

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"Encoder_input": anData[:1000000].astype(np.float32)},
        y=anData_d[:1000000],
        num_epochs=1,
        shuffle=False)
    """c"""

    est_model.train(input_fn=train_input_fn, steps=20000)


    return model
"""c"""


token_ids : tf.Tensor = tf.placeholder(tf.int32, [128, 20])

W : tf.Variable = tf.Variable(tf.zeros([10000, 512]))

token_embedding : tf.Tensor = tf.gather(W, token_ids)      

y_ : tf.Tensor = tf.placeholder(tf.float32, [128, 20, 512])

loss : tf.Tensor = tf.reduce_mean((token_embedding - y_) ** 2)

train_step : tf.Operation = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init : tf.Operation = tf.global_variables_initializer()

loss_summary : tf.Tensor = tf.summary.scalar('mse_loss', loss)



DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE



sess = tf.Session()
sess.run(init)

merged : tf.Tensor = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(DATA_DIR + 'log3')

train_writer.add_graph(sess.graph)




X = np.random.randint(0, 10000, (128, 20)).astype(np.int32)
y = np.random.rand(128, 20, 512)


r = sess.run([train_step, merged], feed_dict={token_ids: X, y_: y})
train_writer.add_summary(r[1], 0)

r = sess.run([train_step, merged], feed_dict={token_ids: X, y_: y})
train_writer.add_summary(r[1], 1)

r = sess.run([train_step, merged], feed_dict={token_ids: X, y_: y})
train_writer.add_summary(r[1], 2)

sess.close()








anData.shape

    iterator = ds.make_initializable_iterator()

    sess.run(iterator.initializer, feed_dict={encoded_placeholder: anData, raw_placeholder: anData_d})
    
    encoded, raw = iterator.get_next()

    return encoded, raw




# Gather experiments.
import numpy as np
import tensorflow as tf

data = np.reshape(np.arange(30), [5, 6])

x = tf.constant(data)

result = tf.gather_nd(x, [3, 5])

sess = tf.Session()

sess.run(result)



:\Users\T149900\AppData\Local\Continuum\anaconda3\lib\site-packages\tensorflow\python\ops\gradients_impl.py:108:

UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. 

This may consume a large amount of memory.
 





Got this warning first time after moving from pure keras to tf.keras.
Based on the above discussion in stackoverflow, looks like tf.keras code has to handle this.


https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation




https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

basic tensorflow graph

