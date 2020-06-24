import sys

sys.path.append('../libs/psm/')
from setup_general_xunet import *

PI = math.pi


# DCT layer process
def DCT_layer(Input):
    DCTBase = np.zeros([4, 4, 1, 16], dtype=np.float32)  # [height,width,input,output]
    u = np.ones([4], dtype=np.float32) * math.sqrt(2.0 / 4.0)
    u[0] = math.sqrt(1.0 / 4.0)
    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                for l in range(0, 4):
                    DCTBase[i, j, :, k * 4 + l] = u[k] * u[l] * math.cos(PI / 8.0 * k * (2 * i + 1)) * math.cos(
                        PI / 8.0 * l * (2 * j + 1))

    DCTKernel = tf.Variable(DCTBase, name="DCTKenel", trainable=False)
    DCT = tf.abs(tf.nn.conv2d(Input, DCTKernel, [1, 1, 1, 1], 'VALID', name="DCT"))
    DCT_Trunc = -(tf.nn.relu(-DCT + 8) - 8)  # Trancation operation
    return DCT_Trunc


# Xu network structure
def conv_layer(input, channels_in, channels_out, istrain, name="layer"):
    with tf.name_scope(name):
        kernel = tf.Variable(tf.random_normal([3, 3, channels_in, channels_out], mean=0.0, stddev=0.01),
                             name="kernel")  # [height,width,input,output]
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME', name="conv")
        bn = tf.layers.batch_normalization(conv, training=istrain)
        relu = tf.nn.relu(bn, name="relu")
        tf.summary.histogram("kernels", kernel)
        return relu


def conv_layer_s(input, channels_in, channels_out, istrain, name="layer_s"):
    with tf.name_scope(name):
        kernel = tf.Variable(tf.random_normal([3, 3, channels_in, channels_out], mean=0.0, stddev=0.01),
                             name="kernel")  # [height,width,input,output]
        conv = tf.nn.conv2d(input, kernel, [1, 2, 2, 1], padding='SAME', name="conv")
        bn = tf.layers.batch_normalization(conv, training=istrain)
        tf.summary.histogram("kernels", kernel)
        return bn


def conv_layer_m(input, bn_s, channels_in, channels_out, step, istrain, name="layer_m"):
    with tf.name_scope(name):
        kernel = tf.Variable(tf.random_normal([3, 3, channels_in, channels_out], mean=0.0, stddev=0.01),
                             name="kernel")  # [height,width,input,output]
        conv = tf.nn.conv2d(input, kernel, [1, step, step, 1], padding='SAME', name="conv")
        bn = tf.layers.batch_normalization(conv, training=istrain)
        relu = tf.nn.relu(bn + bn_s, name="relu")
        tf.summary.histogram("kernels", kernel)
        return relu


# the network structure
def conv_block(input, channels_in, channels_out_1, channels_out_2, istrain, name="block"):
    with tf.name_scope(name):
        relu_1 = conv_layer(input, channels_in, channels_out_1, istrain, name="layer_1")
        bn_s = conv_layer_s(input, channels_in, channels_out_2, istrain, name="layer_s")
        relu_2 = conv_layer_m(relu_1, bn_s, channels_out_1, channels_out_2, 2, istrain, name="layer_m_1")
        relu_3 = conv_layer(relu_2, channels_out_2, channels_out_2, istrain, name="layer_2")
        relu_4 = conv_layer_m(relu_3, relu_2, channels_out_2, channels_out_2, 1, istrain, name="layer_m_2")
        return relu_4


def dense_layer(input, name="dense"):
    with tf.name_scope(name):
        pool_shape = input.get_shape().as_list()
        print 'dense_layer:', input
        # pool_reshape = tf.reshape(input, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        pool_reshape = tf.layers.flatten(input)
        weights = tf.Variable(
            tf.random_normal([pool_shape[1] * pool_shape[2] * pool_shape[3], 2], mean=0.0, stddev=0.01),
            name="weights")
        bias = tf.Variable(tf.random_normal([2], mean=0.0, stddev=0.01), name="bias")
        y_ = tf.matmul(pool_reshape, weights)
        y_ = tf.add(y_, bias, name="y_")
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("bias", bias)
        return y_


class XuNet2(Model):
    def _build_model(self, inputs):
        self.inputs = inputs
        print inputs
        res = DCT_layer(inputs)
        res = conv_block(res, 16, 12, 24, True, name="block_1")
        res = conv_block(res, 24, 24, 48, True, name="block_2")
        res = conv_block(res, 48, 48, 96, True, name="block_3")
        res = conv_block(res, 96, 96, 192, True, name="block_4")
        res = conv_block(res, 192, 192, 384, True, name="block_5")
        res = tf.nn.avg_pool(res, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="VALID",
                             name="pool")  # [input,height,width,ouput]
        y_ = dense_layer(res, name="dense")
        self.outputs = y_
        return self.outputs
