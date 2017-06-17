import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


def three_layers_ann():
    iris = datasets.load_iris()
    # print(iris)
    # print(len(iris['target']))
    training_x = iris['data']
    test_x = iris['data']
    training_y = iris['target']
    test_y = iris['target']
    # length of the input layer
    I = 4
    # length of the hidden layer
    H = 6
    # length of the output layer
    O = 3
    X = tf.placeholder(dtype=tf.float32, shape=[1, I])
    # if the target value being 2, Y_ will be [0,0,1,0]
    # if the target value being 1,Y_ will be [0,1,0,0]
    Y_ = tf.placeholder(dtype=tf.float32, shape=[1, O])
    W1 = tf.Variable(tf.random_uniform([I, H], -1.0, 1.0))
    B1 = tf.Variable(tf.zeros([H]))
    W2 = tf.Variable(tf.random_uniform([H, O], -1.0, 1.0))
    B2 = tf.Variable(tf.zeros([O]))
    Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
    Y2 = tf.nn.softmax(tf.matmul(Y1, W2) + B2)
    cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y2))
    learning_rate = 0.50
    correct_prediction = tf.equal(tf.argmax(Y2, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('begin training process...')
        for step in range(100):
            for i in range(0, len(training_x)):
                if i%25 > 0:
                    continue
                vec_y = np.zeros([1, O], dtype='float32')
                vec_y[0][training_y[i]] = 1.0
                print(vec_y)
                vec_x = training_x[i]
                vec_x = np.reshape(vec_x, (1, 4))
                # vec_y.assign()
                sess.run([optimizer],feed_dict={X: vec_x, Y_: vec_y})
        print('finish training process!')
        _a = 0.0
        cnt = 0
        correct = 0
        for i in range(len(training_x)):
            if i%25 ==0:
                continue
            vec_y = np.zeros([1, O], dtype='float32')
            # vec_y = np.reshape(vec_y, (1, 4))
            vec_y[0][test_y[i]] = 1.0
            vec_x = test_x[i]
            cnt += 1
            vec_x = np.reshape(vec_x, (1, 4))
            _c,_a = sess.run([correct_prediction, accuracy], feed_dict={X: vec_x, Y_: vec_y})
            print(_c)
            print(type(_c))
            if _c[0] == True:
                correct += 1
            print(_c)
        print('*********', _a)
        print('accuracy of test data:', correct*1.0/cnt)


def iris_three_layers_ann_version2():

    """A simple neural network for classify iris data set
     1. using a batch of data for testing and learning
     instead of one singe value per time
     2. introduce to TensorBoard, for visualization of parameters
    """

    iris = datasets.load_iris()
    training_data = {'x':[], 'y':[]}
    test_data = {'x':[], 'y':[]}
    I = 4  # size of input layer
    H = 6  # size of hidden layer
    O = 3  # size of output layer
    batch_size = 75

    def init_data():
        for i in range(0, len(iris['data'])):
            # print(iris['data'][i])
            tmp = None
            if i < 25 or (i < 75 and i >= 50) or (i<125 and i >= 100):
                training_data['x'].append(iris['data'][i])
                tmp = [0.0, 0.0, 0.0]
                target = iris['target'][i]
                tmp[target] = 1.0
                training_data['y'].append(tmp)
            else:
                test_data['x'].append(iris['data'][i])
                tmp = [0.0, 0.0, 0.0]
                target = iris['target'][i]
                tmp[target] = 1.0
                test_data['y'].append(tmp)

        training_data['x'] = np.array(training_data['x'], dtype='float32')
        training_data['y'] = np.array(training_data['y'], dtype='float32')
        training_data['x'] = np.reshape(training_data['x'], [batch_size, I])
        training_data['y'] = np.reshape(training_data['y'], [batch_size, O])
        test_data['x'] = np.reshape(test_data['x'], [batch_size, I])
        test_data['y'] = np.reshape(test_data['y'], [batch_size, O])

    init_data()
    X = tf.placeholder(dtype=tf.float32, shape=[None, I])  # a place for store the input x data
    # Y_ stores the correct y data, not the result we predict
    Y_ = tf.placeholder(dtype=tf.float32, shape=[None, O])
    W1 = tf.Variable(initial_value=tf.random_uniform([I, H], -1.0, 1.0))
    B1 = tf.Variable(initial_value=tf.zeros([H]))
    W2 = tf.Variable(initial_value=tf.random_uniform([H, O], -1.0, 1.0))
    B2 = tf.Variable(initial_value=tf.zeros([O]))
    Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
    Y2 = tf.nn.softmax(tf.matmul(Y1, W2) + B2)
    cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y2)) * O * batch_size
    optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y2, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    def training_process(step):
        # print(training_data['x'])
        o,c,a = sess.run([optimizer, correct_prediction, accuracy], feed_dict={X: training_data['x'], Y_: training_data['y']})
        print('training process running, step: %d' % step)
        print('current accuracy: ', a)


    steps = 200
    for i in range(steps):
        training_process(i)

    # testing process:
    _c,_a= sess.run([correct_prediction, accuracy], feed_dict={X: test_data['x'], Y_: test_data['y']})
    print('accuracy value: ', _a)
    # print(_c)
    sess.close()

if __name__ == '__main__':
    # three_layers_ann()
    iris_three_layers_ann_version2()