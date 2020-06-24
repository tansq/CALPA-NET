# @Time    : 2019-05-18 19:19
# @Author  : wuwl
# @File    : detect_xuNet_test.py
# @Function:


from setup_general_xunet import *
import sys


# detect for XuNet
def get_thinet_input(model_class, thinet_gen, load_path, index_of_block, index_of_layer, batch_size):
    tf.reset_default_graph()
    _runner = GeneratorRunner(thinet_gen, batch_size)
    img_batch, label_batch = _runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NHWC')
    model._build_model(img_batch)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        _runner.start_threads(sess, 1)
        # Input_data = sess.run(get_a_tensor_by_name('Layer' + str(layer) + '/Relu:0'))
        op_Input = sess.run(
            get_a_tensor_by_name('block_' + str(index_of_block) + '/layer_' + str(index_of_layer) + '/relu:0'))
    return op_Input


def get_a_tensor_by_name(name_of_var):
    graph = tf.get_default_graph()
    tar_var_tensor = graph.get_tensor_by_name(name_of_var)
    return tar_var_tensor


def write_2_config_file(index_block, index_layer, acc, rate, config_log, test_acc):
    test_cfg = config_log
    config_raw = ConfigParser.RawConfigParser()
    config_raw.read(test_cfg)
    print 'rate:', rate, '  acc: ', acc
    last_acc = config_raw.getfloat('acc_block' + index_block, 'layer' + index_layer)
    # It can be modified according to the actual situation
    if last_acc <= acc or abs(test_acc - acc) <= 0.05:
        config_raw.remove_option('acc_block' + index_block, 'layer' + index_layer)
        config_raw.set('acc_block' + index_block, 'layer' + index_layer, acc)
        config_raw.remove_option('rate_block' + index_block, 'layer' + index_layer)
        config_raw.set('rate_block' + index_block, 'layer' + index_layer, rate)
        config_raw.write(open(config_log, 'w'))


def thinet_detect_on_cov_layer(model_class, index_of_block, index_of_layer, gen, batch_size, ds_size, thinet_gen,
                               load_path, f_log, config_log, test_acc):
    Input_data = get_thinet_input(model_class, thinet_gen, load_path, index_of_block, index_of_layer, batch_size)
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    # img_batch, label_batch = read_a_batch_from_Input_dir(batch_size, thinet_gen)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NHWC')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss', \
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy', \
                                       float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op, \
                            accuracy_summary.increment_op)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)
        f_log.write('-----------block ' + str(index_of_block) + ' layer ' + str(index_of_layer) + '-------------\n')

        next_pruned_kernel = get_a_tensor_by_name(
            'block_' + str(index_of_block) + '/layer_m_' + str(index_of_layer) + '/kernel:0')
        pruned_kernel = get_a_tensor_by_name(
            'block_' + str(index_of_block) + '/layer_' + str(index_of_layer) + '/kernel:0')
        BN_index = 5 * (index_of_block - 1) + 3 * (index_of_layer - 1)
        if BN_index == 0:
            mean = get_a_tensor_by_name('batch_normalization/moving_mean:0')
            variance = get_a_tensor_by_name('batch_normalization/moving_variance:0')
            beta = get_a_tensor_by_name('batch_normalization/beta:0')
            gamma = get_a_tensor_by_name('batch_normalization/gamma:0')
        else:
            mean = get_a_tensor_by_name('batch_normalization_' + str(BN_index) + '/moving_mean:0')
            variance = get_a_tensor_by_name('batch_normalization_' + str(BN_index) + '/moving_variance:0')
            beta = get_a_tensor_by_name('batch_normalization_' + str(BN_index) + '/beta:0')
            gamma = get_a_tensor_by_name('batch_normalization_' + str(BN_index) + '/gamma:0')

        next_pruned_kernel_data, pruned_kernel_data, mean_data, variance_data, beta_data, gamma_data = sess.run(
            [next_pruned_kernel, pruned_kernel, mean, variance, beta, gamma])
        # feed_dict={x: img_batch, y: label_batch, is_training: False}

        # greedy algorithm
        with tf.name_scope("copy"):
            input = tf.placeholder(tf.float32, shape=[None, None, None, None], name='None_input')
            if (index_of_layer == 1):
                kernel_name = 'block_' + str(index_of_block) + '/layer_m_1/kernel:0'
                # with tf.Session() as sess:
                kernel_data = sess.run(get_a_tensor_by_name(kernel_name))
                kernel = tf.Variable(kernel_data, name="kernel")
                conv = tf.nn.conv2d(input, kernel, [1, 2, 2, 1], padding='SAME', name="conv")
            if (index_of_layer == 2):
                kernel_name = 'block_' + str(index_of_block) + '/layer_m_2/kernel:0'
                # with tf.Session() as sess:
                kernel_data = sess.run(get_a_tensor_by_name(kernel_name))
                kernel = tf.Variable(kernel_data, name="kernel")
                conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME', name="conv")
            sub_of_kernel = tf.placeholder(tf.float32, shape=[None, None, None, None], name="sub_of_kernel")
            op_ass = tf.assign(kernel, sub_of_kernel, validate_shape=False, name="op_ass")

        channel_to_remove = []

        S = []
        for i in range(Input_data.shape[3]):
            if i not in channel_to_remove:
                S.append(i)
        next_kernel_selected = next_pruned_kernel_data[:, :, channel_to_remove, :]
        Input_selected = Input_data[:, :, :, channel_to_remove]

        step = Input_data.shape[3] * 0.05
        step_count = 1  # 1-19
        while len(channel_to_remove) < Input_data.shape[3]:
            min = sys.maxsize
            min_index = -1
            for i in S:
                channel_to_remove.append(i)
                sess.run(op_ass, feed_dict={
                    sub_of_kernel: np.insert(next_kernel_selected, channel_to_remove.__len__() - 1,
                                             next_pruned_kernel_data[:, :, i, :], 2)})
                cur_output = sess.run(conv, feed_dict={
                    input: np.insert(Input_selected, channel_to_remove.__len__() - 1, Input_data[:, :, :, i],
                                     3)})
                if np.mean(np.square(cur_output)) < min:
                    min = np.mean(np.square(cur_output))
                    min_index = i
                channel_to_remove.pop()
            channel_to_remove.append(min_index)
            S.remove(min_index)
            next_kernel_selected = np.insert(next_kernel_selected, channel_to_remove.__len__() - 1,
                                             next_pruned_kernel_data[:, :, min_index, :], 2)
            Input_selected = np.insert(Input_selected, channel_to_remove.__len__() - 1,
                                       Input_data[:, :, :, min_index], 3)

            if len(channel_to_remove) >= step * step_count:
                print(channel_to_remove)

                # add at 5.19 11:54
                S = []
                for i in range(Input_data.shape[3]):
                    if i not in channel_to_remove:
                        S.append(i)

                sess.run(tf.assign(pruned_kernel, pruned_kernel_data[:, :, :, S], validate_shape=False))
                sess.run(tf.assign(next_pruned_kernel, next_pruned_kernel_data[:, :, S, :], validate_shape=False))
                sess.run(tf.assign(mean, mean_data[S], validate_shape=False))
                sess.run(tf.assign(variance, variance_data[S], validate_shape=False))
                sess.run(tf.assign(beta, beta_data[S], validate_shape=False))
                sess.run(tf.assign(gamma, gamma_data[S], validate_shape=False))
                for j in range(0, ds_size, batch_size):
                    sess.run(increment_op)
                mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable, \
                                                     accuracy_summary.mean_variable])
                sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])

                f_log.write('rate ' + str(0.05 * step_count) + ',acc ' + str(mean_accuracy) + ',loss ' + str(
                    mean_loss) + '\n')
                f_log.flush()
                f_log.write(str(S) + '\n')
                write_2_config_file(str(index_of_block), str(index_of_layer), mean_accuracy, str(0.05 * step_count),
                                    config_log, test_acc)
                step_count = step_count + 1


# xunet Li detect
def Li_detect_on_conv_layer(model_class, index_of_block, index_of_layer, gen, batch_size, ds_size,
                            load_path, f_log, config_log, test_acc):
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NHWC')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss', \
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy', \
                                       float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op, \
                            accuracy_summary.increment_op)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)
        f_log.write('----------- block ' + str(index_of_block) + ' layer ' + str(index_of_layer) + '-------------\n')

        print 'block:', str(index_of_block)
        BN_index_layer_s = 5 * (index_of_block - 1) + 1
        BN_index_layer_m1 = 5 * (index_of_block - 1) + 2
        BN_index_layer_m2 = 5 * (index_of_block - 1) + 4

        # 'block_' + str(index_of_block) + '/layer_m_' + str(index_of_layer)
        s_kernel_tensor = get_a_tensor_by_name('block_' + str(index_of_block) + '/layer_s' + '/kernel:0')
        # s_bias_tensor = get_a_tensor_by_name('block_' + str(index_of_block) + '/layer_s' + '/kernel:0')
        s_mean_tensor = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_s) + '/moving_mean:0')
        s_var_tensor = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_s) + '/moving_variance:0')
        s_gamma_tensor = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_s) + '/gamma:0')
        s_beta_tensor = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_s) + '/beta:0')

        # S = []
        _S = []

        if index_of_layer == 1:
            _kernel_tensor = get_a_tensor_by_name('block_' + str(index_of_block) + '/layer_m_1' + '/kernel:0')
            # _bias_tensor = get_a_tensor_by_name('block_' + str(index_of_block) + '/layer_m_1' + '/kernel:0')
            _mean_tensor = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_m1) + '/moving_mean:0')
            _var_tensor = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_m1) + '/moving_variance:0')
            _gamma_tensor = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_m1) + '/gamma:0')
            _beta_tensor = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_m1) + '/beta:0')

            _kernel_tensor_ = get_a_tensor_by_name('block_' + str(index_of_block) + '/layer_m_2' + '/kernel:0')
            # _bias_tensor = get_a_tensor_by_name('block_' + str(index_of_block) + '/layer_m_1' + '/kernel:0')
            _mean_tensor_ = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_m2) + '/moving_mean:0')
            _var_tensor_ = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_m2) + '/moving_variance:0')
            _gamma_tensor_ = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_m2) + '/gamma:0')
            _beta_tensor_ = get_a_tensor_by_name('batch_normalization_' + str(BN_index_layer_m2) + '/beta:0')

            next_kernel_tensor_for_m1 = get_a_tensor_by_name(
                'block_' + str(index_of_block) + '/layer_' + str(index_of_layer + 1) + '/kernel:0')
            if index_of_block < 5:
                next_kernel_tensor_for_m2 = get_a_tensor_by_name(
                    'block_' + str(index_of_block + 1) + '/layer_' + str(index_of_layer) + '/kernel:0')
                next_kernel_tensor_for_m2_s = get_a_tensor_by_name(
                    'block_' + str(index_of_block + 1) + '/layer_s' + '/kernel:0')
            if index_of_block == 5:
                next_weights_tensor = get_a_tensor_by_name('dense/weights:0')
                next_pool_tensor = get_a_tensor_by_name('pool:0')


            s_kernel_data, s_mean_data, s_var_data, s_gamma_data, s_beta_data, \
            _kernel_data, _mean_data, _var_data, _gamma_data, _beta_data, _kernel_data_, \
            _mean_data_, _var_data_, _gamma_data_, _beta_data_, \
            next_kernel_data_for_m1 = sess.run([
                s_kernel_tensor, s_mean_tensor, s_var_tensor, s_gamma_tensor, s_beta_tensor,
                _kernel_tensor, _mean_tensor, _var_tensor, _gamma_tensor, _beta_tensor,
                _kernel_tensor_, _mean_tensor_, _var_tensor_, _gamma_tensor_, _beta_tensor_,
                next_kernel_tensor_for_m1])

            if index_of_block < 5:
                next_kernel_data_for_m2, next_kernel_data_for_m2_s = sess.run(
                    [next_kernel_tensor_for_m2, next_kernel_tensor_for_m2_s])
            # if index_of_layer != 11:
            #     next_s_kernel_data, next_s_bias_data = sess.run([next_s_kernel_tensor, next_s_bias_tensor])
            if index_of_block == 5:
                next_weights_data = sess.run([next_weights_tensor])
                print 'next weights data ', len(next_weights_data[0])
                next_pool_data = sess.run([next_pool_tensor])
                print 'next pool data ', len(next_pool_data[0])

            shortcut_kernel = s_kernel_data
            shortcut_kernel_abs = np.abs(shortcut_kernel)
            shape = shortcut_kernel.shape
            channal_L1 = []
            for channal in range(shape[3]):
                channal_L1.append(np.sum(shortcut_kernel_abs[:, :, :, channal]))
            channal_L1 = np.array(channal_L1)
            channal_L1_sorted_index = np.argsort(-channal_L1).tolist()
            for ratio in range(1, 20):
                lens = int((20 - ratio) * 0.05 * shape[3])
                S = channal_L1_sorted_index[0:lens]
                print S
                sess.run(tf.assign(s_kernel_tensor, s_kernel_data[:, :, :, S], validate_shape=False))
                # sess.run(tf.assign(s_bias_tensor, s_bias_data[S], validate_shape=False))
                sess.run(tf.assign(s_mean_tensor, s_mean_data[S], validate_shape=False))
                sess.run(tf.assign(s_var_tensor, s_var_data[S], validate_shape=False))
                sess.run(tf.assign(s_gamma_tensor, s_gamma_data[S], validate_shape=False))
                sess.run(tf.assign(s_beta_tensor, s_beta_data[S], validate_shape=False))

                sess.run(tf.assign(_kernel_tensor, _kernel_data[:, :, :, S], validate_shape=False))
                # sess.run(tf.assign(_bias_tensor, _bias_data[S], validate_shape=False))
                sess.run(tf.assign(_mean_tensor, _mean_data[S], validate_shape=False))
                sess.run(tf.assign(_var_tensor, _var_data[S], validate_shape=False))
                sess.run(tf.assign(_gamma_tensor, _gamma_data[S], validate_shape=False))
                sess.run(tf.assign(_beta_tensor, _beta_data[S], validate_shape=False))

                sess.run(tf.assign(_kernel_tensor_, _kernel_data_[:, :, :, S], validate_shape=False))
                # sess.run(tf.assign(_bias_tensor, _bias_data[S], validate_shape=False))
                sess.run(tf.assign(_mean_tensor_, _mean_data_[S], validate_shape=False))
                sess.run(tf.assign(_var_tensor_, _var_data_[S], validate_shape=False))
                sess.run(tf.assign(_gamma_tensor_, _gamma_data_[S], validate_shape=False))
                sess.run(tf.assign(_beta_tensor_, _beta_data_[S], validate_shape=False))

                sess.run(
                    tf.assign(next_kernel_tensor_for_m1, next_kernel_data_for_m1[:, :, S, :], validate_shape=False))
                if index_of_block < 5:
                    sess.run(
                        tf.assign(next_kernel_tensor_for_m2, next_kernel_data_for_m2[:, :, S, :], validate_shape=False))
                    sess.run(tf.assign(next_kernel_tensor_for_m2_s, next_kernel_data_for_m2_s[:, :, S, :],
                                       validate_shape=False))
                if index_of_block == 5:
                    sess.run(tf.assign(next_weights_tensor, next_weights_data[0][S], validate_shape=False))

                for j in range(0, ds_size, batch_size):
                    sess.run(increment_op)
                mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable, \
                                                     accuracy_summary.mean_variable])
                sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])
                f_log.write(
                    'rate ' + str(ratio * 0.05) + ',acc ' + str(mean_accuracy) + ',loss ' + str(mean_loss) + '\n')
                f_log.flush()
                f_log.write('layer_S:' + str(S) + '\n')
                f_log.flush()
                write_2_config_file(str(index_of_block), str(index_of_layer), mean_accuracy, str(0.05 * ratio),
                                    config_log, test_acc)

        else:
            print "wrong layer index!"

