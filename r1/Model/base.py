import tensorflow as tf

class Base():
    def save_model(sess, path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)

    def restore_model(sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path)

    def weight_varible(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)