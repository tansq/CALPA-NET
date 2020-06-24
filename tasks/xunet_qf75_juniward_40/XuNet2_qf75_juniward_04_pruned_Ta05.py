# @Time    : 2019-05-22 20:06
# @Author  : wuwl
# @File    : XuNet2_qf75_juniward_04_pruned_Ta05.py
# @Function: Construction of Network structure after pruning

from setup_xunet_qf75_juniward_04 import *

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
def conv_block(input, channels_in, channels_out_1, channels_out_2, ratio1, ratio2, ratio3, istrain, name="block"):
    with tf.name_scope(name):
        relu_1 = conv_layer(input, channels_in, int(channels_out_1 * (1 - ratio1)), istrain, name="layer_1")  # thinet
        bn_s = conv_layer_s(input, channels_in, int(channels_out_2 * (1 - ratio3)), istrain, name="layer_s")  # li
        relu_2 = conv_layer_m(relu_1, bn_s, int(channels_out_1 * (1 - ratio1)), int(channels_out_2 * (1 - ratio3)), 2,
                              istrain,  # li
                              name="layer_m_1")
        relu_3 = conv_layer(relu_2, int(channels_out_2 * (1 - ratio3)), int(channels_out_2 * (1 - ratio2)), istrain,
                            # thinet
                            name="layer_2")
        relu_4 = conv_layer_m(relu_3, relu_2, int(channels_out_2 * (1 - ratio2)), int(channels_out_2 * (1 - ratio3)), 1,
                              istrain,  # li
                              name="layer_m_2")
        return relu_4


def dense_layer(input, name="dense"):
    with tf.name_scope(name):
        pool_shape = input.get_shape().as_list()
        #pool_reshape = tf.reshape(input, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
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


def fill_pruned_thinet_rate(pruned_log):
    config_raw = ConfigParser.RawConfigParser()
    config_raw.read(pruned_log)
    list_rates = []
    for i in range(2, 6):  # skip rate_block1
        for j in range(1, 3):
            list_rates.append(config_raw.getfloat('rate_block' + str(i), 'layer' + str(j)))
    return list_rates

def fill_pruned_li_rate(pruned_log_li):
    config_raw = ConfigParser.RawConfigParser()
    config_raw.read(pruned_log_li)
    list_rates = []
    for i in range(2, 6):  # skip rate_block1
        list_rates.append(config_raw.getfloat('rate_block' + str(i), 'layer1'))
    return list_rates

class XuNet2_pruned(Model):
    def _build_model(self, inputs):
        self.inputs = inputs
        res = DCT_layer(inputs)
        thinet_pruned_rates = fill_pruned_thinet_rate(config_log_05_thinet)
        li_pruned_rates = fill_pruned_li_rate(config_log_05_li)
        res = conv_block(res, 16, 12, 24, 0, 0, 0, True, name="block_1")
        res = conv_block(res, 24, 24, 48, thinet_pruned_rates[0], thinet_pruned_rates[1], li_pruned_rates[0], True, name="block_2")
        res = conv_block(res, int(48 * (1 - li_pruned_rates[0])), 48, 96, thinet_pruned_rates[2], thinet_pruned_rates[3], li_pruned_rates[1], True, name="block_3")
        res = conv_block(res, int(96 * (1 - li_pruned_rates[1])), 96, 192, thinet_pruned_rates[4], thinet_pruned_rates[5], li_pruned_rates[2], True, name="block_4")
        res = conv_block(res, int(192 * (1 - li_pruned_rates[2])), 192, 384, thinet_pruned_rates[6], thinet_pruned_rates[7], li_pruned_rates[3], True, name="block_5")
        res = tf.nn.avg_pool(res, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="VALID",
                             name="pool")  # [input,height,width,output]
        y_ = dense_layer(res, name="dense")
        self.outputs = y_
        return self.outputs