import tensorflow as tf
import  numpy as np

def mul_test():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    sess = tf.Session()
    result = sess.run(product)
    print(result)
def shape_test():
    a = np.array([1, 2, 3, 4, 5, 6])
    print(a.shape)

if __name__ == '__main__':
    shape_test()