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
    sess.close()


def matmul_traditional_python_codes():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 2], [3, 4]])
    C = np.dot(A, B)
    print(C)

def build_our_graph():

    """
    given matrix A,B,D,
    hence C = A * B, E = C * D
    now we want to compute E
    :return:
    """
    _A = np.array([[1, 2], [3, 4]], dtype=np.int32)
    _B = np.array([[1, 0], [0, 1]], dtype=np.int32)
    _D = np.array([[1, 0], [0, 1]], dtype=np.int32)
    A = tf.constant(_A)
    B = tf.constant(_B)
    D = tf.constant(_D)
    C = tf.matmul(A, B)
    print('type of matrix A: %s' %type(A))
    print('type of matrix C: %s' %type(C))
    E = tf.matmul(C, D)
    with tf.Session() as sess:
        # ans = sess.run(E)
        ans = sess.run([C, E])
    print(ans)

if __name__ == '__main__':
    # multiply_B_to_A()
    # matmul_traditional_python_codes()
    build_our_graph()