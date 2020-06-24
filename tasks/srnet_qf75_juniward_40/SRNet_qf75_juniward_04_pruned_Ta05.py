# @Time    : 2019-05-24 22:15
# @Author  : WeiLong Wu
# @File    : SRNet_qf75_juniward_04_pruned_Ta05.py
# @Function: Construction of Network structure after pruning


from setup_srnet_qf75_juniward_04 import *

def fill_pruned_rate(pruned_log):
    config_raw = ConfigParser.RawConfigParser()
    config_raw.read(pruned_log)
    Thinet_detect_rates = []
    L1_detect_rates = []
    for i in range(3, 13):
        Thinet_detect_rates.append(1 - config_raw.getfloat('Thinet_rate', 'layer' + str(i)))
    for j in range(8, 13):
        L1_detect_rates.append(1 - config_raw.getfloat('L1_rate', 'layer' + str(j)))
    return Thinet_detect_rates, L1_detect_rates

class SRNet_pruned(Model):
    def _build_model(self, inputs):
        self.inputs = inputs
        thinet_save_rates, li_save_rates = fill_pruned_rate(config_log_05)
        if self.data_format == 'NCHW':
            reduction_axis = [2, 3]
            _inputs = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
        else:
            reduction_axis = [1, 2]
            _inputs = tf.cast(inputs, tf.float32)
        with arg_scope([layers.conv2d], num_outputs=16,
                       kernel_size=3, stride=1, padding='SAME',
                       data_format=self.data_format,
                       activation_fn=None,
                       weights_initializer=layers.variance_scaling_initializer(),
                       weights_regularizer=layers.l2_regularizer(2e-4),
                       biases_initializer=tf.constant_initializer(0.2),
                       biases_regularizer=None), \
             arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True,
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format), \
             arg_scope([layers.avg_pool2d],
                       kernel_size=[3, 3], stride=[2, 2], padding='SAME',
                       data_format=self.data_format):
            with tf.variable_scope('Layer1'):
                conv = layers.conv2d(_inputs, num_outputs=64, kernel_size=3)
                actv = tf.nn.relu(layers.batch_norm(conv))
            with tf.variable_scope('Layer2'):
                conv = layers.conv2d(actv)
                actv = tf.nn.relu(layers.batch_norm(conv))
            with tf.variable_scope('Layer3'):
                conv1 = layers.conv2d(actv, num_outputs=int(16 * thinet_save_rates[0]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(actv, bn2)
            with tf.variable_scope('Layer4'):
                conv1 = layers.conv2d(res, num_outputs=int(16 * thinet_save_rates[1]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(res, bn2)
            with tf.variable_scope('Layer5'):
                conv1 = layers.conv2d(res, num_outputs=int(16 * thinet_save_rates[2]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                res = tf.add(res, bn)
            with tf.variable_scope('Layer6'):
                conv1 = layers.conv2d(res, num_outputs=int(16 * thinet_save_rates[3]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                res = tf.add(res, bn)
            with tf.variable_scope('Layer7'):
                conv1 = layers.conv2d(res, num_outputs=int(16 * thinet_save_rates[4]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                res = tf.add(res, bn)
            with tf.variable_scope('Layer8'):
                convs = layers.conv2d(res, num_outputs=int(16 * li_save_rates[0]), kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=int(16 * thinet_save_rates[5]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=int(16 * li_save_rates[0]))
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer9'):
                convs = layers.conv2d(res, num_outputs=int(64 * li_save_rates[1]), kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=int(64 * thinet_save_rates[6]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=int(64 * li_save_rates[1]))
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer10'):
                convs = layers.conv2d(res, num_outputs=int(128 * li_save_rates[2]), kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=int(128 * thinet_save_rates[7]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=int(128 * li_save_rates[2]))
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer11'):
                convs = layers.conv2d(res, num_outputs=int(256 * li_save_rates[3]), kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=int(256 * thinet_save_rates[8]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=int(256 * li_save_rates[3]))
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer12'):
                conv1 = layers.conv2d(res, num_outputs=int(512 * thinet_save_rates[9]))
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=int(512 * li_save_rates[4]))
                bn = layers.batch_norm(conv2)
                avgp = tf.reduce_mean(bn, reduction_axis, keep_dims=True)
        ip = layers.fully_connected(layers.flatten(avgp), num_outputs=2,
                                    activation_fn=None, normalizer_fn=None,
                                    weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                    biases_initializer=tf.constant_initializer(0.), scope='ip')
        self.outputs = ip
        return self.outputs
