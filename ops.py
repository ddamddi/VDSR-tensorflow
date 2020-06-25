import tensorflow as tf

w_initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
w_regularizer = tf.contrib.layers.l2_regularizer(0.0001)

def conv(x, channels, kernel=3, stride=1, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels, strides=stride,
                             kernel_size=kernel, kernel_initializer=w_initializer,
                             kernel_regularizer=w_regularizer,
                             use_bias=use_bias, padding=padding)

        return x

def relu(x):
    return tf.nn.relu(x)