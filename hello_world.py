import tensorflow as tf

def hello_world():
    hello = tf.constant('hello world')
    sess = tf.Session()
    print(sess.run(hello))
    sess.close()

if __name__ == '__main__':
    hello_world()