import tensorflow as tf
import numpy as np

def multiply_B_to_A():
    A = tf.Variable(np.array([[1, 0], [0, 1]]), dtype='int32')
    B = tf.Variable(np.array([[1, 2], [3, 4]]), dtype='int32')
    C = tf.matmul(A, B)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    ans = sess.run(C)
    print(ans)


def matmul_traditional_python_codes():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 2], [3, 4]])
    C = np.dot(A, B)
    print(C)


if __name__ == '__main__':
    # multiply_B_to_A()
    matmul_traditional_python_codes()