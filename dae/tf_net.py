

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


# tf Graph input

X = tf.placeholder("float", [None, n_input])


Y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}


biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model

def multilayer_perceptron(x):
    
    # Hidden fully connected layer with 256 neuronsp
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    
    return out_layer

 
# Construct model

logits = multilayer_perceptron(X)

# Define loss 
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)

# Initializing the variables

init = tf.global_variables_initializer()


sess = tf.Session()

sess.run(init)

# Training cycle
for epoch in range(training_epochs):

    avg_cost = 0.

    total_batch = int(mnist.train.num_examples/batch_size)

    # Loop over all batches
    for i in range(total_batch):
        
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

        # Compute average loss
        avg_cost += c / total_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))



# Test model
pred = tf.nn.softmax(logits)  # Apply softmax to logits

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuracy:", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


import tensorflow as tf

data = loadY()

num_features = data.shape[1]

X = tf.placeholder("float", [None, num_features])

Y = tf.placeholder("float", [None, num_features])


weights = {
    'h0': tf.Variable(tf.random_normal([num_features, 1500])),
    'h1': tf.Variable(tf.random_normal([1500, 1500])),
    'h2': tf.Variable(tf.random_normal([1500, 1500])),
    'out': tf.Variable(tf.random_normal([1500, num_features]))
}

biases = {
    'b0': tf.Variable(tf.random_normal([1500])),
    'b1': tf.Variable(tf.random_normal([1500])),
    'b2': tf.Variable(tf.random_normal([1500])),
    'out': tf.Variable(tf.random_normal([num_features]))
}

def multilayer_perceptron(x):
    
   
    layer_0 = tf.add(tf.matmul(x, weights['h0']), biases['b0'])
    
    layer_1 = tf.add(tf.matmul(layer_0, weights['h1']), biases['b1'])

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    
    return out_layer




my_model = multilayer_perceptron(X)

loss_op = tf.losses.mean_squared_error(labels=Y, predictions=my_model)


# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.003)


train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)


nRows = data.shape[0]


# Create a small batch

batch_x = data[:128]
batch_y = data[:128]



_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

# Compute average loss
avg_cost = c 

print(f"avg_cost = {avg_cost}")




#######################################################################################


