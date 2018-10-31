import tensorflow as tf


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    with tf.name_scope('Reshape'):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

    with tf.name_scope('Conv_1'):
        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

    with tf.name_scope('Conv_2'):
        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

    with tf.name_scope('FC_1'):
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope('Dropout'):
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

    with tf.name_scope('FC_2'):
        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def wrapper(num_classes, keep_prob, learning_rate, X, Y):
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.get_variable("WC1", shape=[5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer()),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.get_variable("WC2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.get_variable("WD1", shape=[7 * 7 * 64, 1024], initializer=tf.contrib.layers.xavier_initializer()),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.get_variable("OUT", shape=[1024, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Construct model
    with tf.name_scope('Model'):
        logits = conv_net(X, weights, biases, keep_prob)

    with tf.name_scope('Predictions'):
        prediction = tf.nn.softmax(logits)

    with tf.name_scope('Loss'):
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        ########################################################################################################################
        # Instead of calculating and applying gradients automatically we are applying then using optimizer.apply_gradients     #
        ########################################################################################################################

        # Op to calculate every variable gradient
        grads = tf.gradients(loss_op, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))
        # Op to update all variables according to their gradient
        train_op = optimizer.apply_gradients(grads_and_vars=grads)

    # train_op = optimizer.minimize(loss_op)

    with tf.name_scope('Accuracy'):
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    return weights, biases, logits, prediction, accuracy, init, loss_op, train_op, grads
