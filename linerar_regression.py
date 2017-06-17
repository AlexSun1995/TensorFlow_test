import tensorflow as tf
import numpy as np
# given a training set (x, y)n,
# build a model y = wx + b

def test_0():
    X = tf.placeholder(dtype='float')
    Y = tf.placeholder(dtype='float')
    W = tf.Variable(np.random.rand(), name='weight')
    b = tf.Variable(np.random.rand(), name='bias')
    # A = WX + b
    activation = tf.add(tf.multiply(W, X), b)
    # cost function
    cost = tf.reduce_mean(tf.square((activation - Y)))
    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    training_x = np.linspace(-1, 1, 100)
    training_y = 2 * training_x + 10
    with tf.Session() as sess:
        sess.run(init)
        for i in range(50):
            for (x, y) in zip(training_x, training_y):
                _, w, b1 = sess.run([optimizer, W, b], feed_dict={X: x, Y: y})
        print('******* this is the weight value:', w)
        print('*******this is the bias value:', b1)


def test_of_higher_dimension():
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    training_x = np.float32(np.random.rand(2, 100))
    training_y = np.dot([0.300, 0.400], training_x) + 0.500
    y = tf.matmul(W, training_x) + b
    # cost function
    cost = tf.reduce_mean(tf.square(y - training_y))
    learning_rate = 0.5
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(200):
            _, _w, _b = sess.run([optimizer, W, b])
            if(step%20 == 0):
                print(step, _w, _b)



if __name__ == '__main__':
    test_of_higher_dimension()

