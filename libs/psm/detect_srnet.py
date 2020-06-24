# @Time    : 2019-05-11 09:59
# @Author  : WeiLong Wu
# @File    : detect_srnet.py
# @Function: Functional module for detect

from setup_general_srnet import *
import sys


# detect for SRNet

# get the input_data from target layer
def get_thinet_input(model_class, thinet_gen, load_path, layer):
    tf.reset_default_graph()
    batch_size = 100
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
        Input_data = sess.run(get_a_tensor_by_name('Layer' + str(layer) + '/Relu:0'))
    return Input_data


def get_a_tensor_by_name(name_of_var):
    graph = tf.get_default_graph()
    tar_var_tensor = graph.get_tensor_by_name(name_of_var)
    return tar_var_tensor


#  It's going to help you extract the list of weights that you have in CKPT
def optimistic_restore_vars(model_ckpt_path):
    reader = tf.train.NewCheckpointReader(model_ckpt_path)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    return restore_vars


def write_2_config_file(detect_type, index_layer, acc, rate, config_log_list, test_acc):
    test_cfg02 = config_log_list[0]
    test_cfg05 = config_log_list[1]
    test_cfg10 = config_log_list[2]
    test_cfg20 = config_log_list[3]
    test_cfg50 = config_log_list[4]

    config_raw02 = ConfigParser.RawConfigParser()
    config_raw02.read(test_cfg02)
    config_raw05 = ConfigParser.RawConfigParser()
    config_raw05.read(test_cfg05)
    config_raw10 = ConfigParser.RawConfigParser()
    config_raw10.read(test_cfg10)
    config_raw20 = ConfigParser.RawConfigParser()
    config_raw20.read(test_cfg20)
    config_raw50 = ConfigParser.RawConfigParser()
    config_raw50.read(test_cfg50)
    print 'rate:', rate, '  acc: ', acc
    # [0.02, 0.05, 0.10, 0.20, 0.50]
    if detect_type == 'Thinet':
        # The last acc recorded was smaller than the current acc
        last_acc = config_raw02.getfloat('Thinet_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.02:
            config_raw02.remove_option('Thinet_rate', 'layer' + index_layer)
            config_raw02.set('Thinet_rate', 'layer' + index_layer, rate)
            config_raw02.remove_option('Thinet_acc', 'layer' + index_layer)
            config_raw02.set('Thinet_acc', 'layer' + index_layer, acc)
            config_raw02.write(open(test_cfg02, 'w'))

        last_acc = config_raw05.getfloat('Thinet_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.05:
            config_raw05.remove_option('Thinet_rate', 'layer' + index_layer)
            config_raw05.set('Thinet_rate', 'layer' + index_layer, rate)
            config_raw05.remove_option('Thinet_acc', 'layer' + index_layer)
            config_raw05.set('Thinet_acc', 'layer' + index_layer, acc)
            config_raw05.write(open(test_cfg05, 'w'))

        last_acc = config_raw10.getfloat('Thinet_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.10:
            config_raw10.remove_option('Thinet_rate', 'layer' + index_layer)
            config_raw10.set('Thinet_rate', 'layer' + index_layer, rate)
            config_raw10.remove_option('Thinet_acc', 'layer' + index_layer)
            config_raw10.set('Thinet_acc', 'layer' + index_layer, acc)
            config_raw10.write(open(test_cfg10, 'w'))

        last_acc = config_raw20.getfloat('Thinet_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.20:
            config_raw20.remove_option('Thinet_rate', 'layer' + index_layer)
            config_raw20.set('Thinet_rate', 'layer' + index_layer, rate)
            config_raw20.remove_option('Thinet_acc', 'layer' + index_layer)
            config_raw20.set('Thinet_acc', 'layer' + index_layer, acc)
            config_raw20.write(open(test_cfg20, 'w'))

        last_acc = config_raw50.getfloat('Thinet_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.50:
            config_raw50.remove_option('Thinet_rate', 'layer' + index_layer)
            config_raw50.set('Thinet_rate', 'layer' + index_layer, rate)
            config_raw50.remove_option('Thinet_acc', 'layer' + index_layer)
            config_raw50.set('Thinet_acc', 'layer' + index_layer, acc)
            config_raw50.write(open(test_cfg50, 'w'))

    if detect_type == 'L1':
        # The last acc recorded was smaller than the current acc
        last_acc = config_raw02.getfloat('L1_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.02:
            config_raw02.remove_option('L1_rate', 'layer' + index_layer)
            config_raw02.set('L1_rate', 'layer' + index_layer, rate)
            config_raw02.remove_option('L1_acc', 'layer' + index_layer)
            config_raw02.set('L1_acc', 'layer' + index_layer, acc)
            config_raw02.write(open(test_cfg02, 'w'))

        last_acc = config_raw05.getfloat('L1_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.05:
            config_raw05.remove_option('L1_rate', 'layer' + index_layer)
            config_raw05.set('L1_rate', 'layer' + index_layer, rate)
            config_raw05.remove_option('L1_acc', 'layer' + index_layer)
            config_raw05.set('L1_acc', 'layer' + index_layer, acc)
            config_raw05.write(open(test_cfg05, 'w'))

        last_acc = config_raw10.getfloat('L1_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.10:
            config_raw10.remove_option('L1_rate', 'layer' + index_layer)
            config_raw10.set('L1_rate', 'layer' + index_layer, rate)
            config_raw10.remove_option('L1_acc', 'layer' + index_layer)
            config_raw10.set('L1_acc', 'layer' + index_layer, acc)
            config_raw10.write(open(test_cfg10, 'w'))

        last_acc = config_raw20.getfloat('L1_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.20:
            config_raw20.remove_option('L1_rate', 'layer' + index_layer)
            config_raw20.set('L1_rate', 'layer' + index_layer, rate)
            config_raw20.remove_option('L1_acc', 'layer' + index_layer)
            config_raw20.set('L1_acc', 'layer' + index_layer, acc)
            config_raw20.write(open(test_cfg20, 'w'))

        last_acc = config_raw50.getfloat('L1_acc', 'layer' + index_layer)
        if last_acc <= acc or abs(test_acc - acc) <= 0.50:
            config_raw50.remove_option('L1_rate', 'layer' + index_layer)
            config_raw50.set('L1_rate', 'layer' + index_layer, rate)
            config_raw50.remove_option('L1_acc', 'layer' + index_layer)
            config_raw50.set('L1_acc', 'layer' + index_layer, acc)
            config_raw50.write(open(test_cfg50, 'w'))


def detect_one_layer_Thinet(model_class, gen, batch_size, ds_size, thinet_gen, load_path, layer, f_log, f_log_S,
                            config_log_list, test_acc):
    Input_data = get_thinet_input(model_class, thinet_gen, load_path, layer)
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NHWC')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss',
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy',
                                       float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op,
                            accuracy_summary.increment_op)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)
        f_log.write('----------- layer ' + str(layer) + '-------------\n')
        f_log_S.write('----------- layer ' + str(layer) + '-------------\n')

        if layer >= 3 and layer <= 7 or layer == 12:
            tar_kernel_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv/weights:0')
            tar_bias_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv/biases:0')
            tar_mean_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm/moving_mean:0')
            tar_var_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm/moving_variance:0')
            tar_gamma_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm/gamma:0')
            tar_beta_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm/beta:0')
            next_bias_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_1/biases:0')
            next_kernel_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_1/weights:0')

            tar_kernel_data, tar_bias_data, tar_mean_data, tar_var_data, tar_gamma_data, tar_beta_data, \
            next_bias_data, next_kernel_data = sess.run([
                tar_kernel_tensor, tar_bias_tensor, tar_mean_tensor, tar_var_tensor, tar_gamma_tensor, tar_beta_tensor, \
                next_bias_tensor, next_kernel_tensor])

            # copy of one conv for greedy iteration
            with tf.variable_scope('Copy'):
                ph_input = tf.placeholder(tf.float32, shape=[None, None, None, None])
                copy_weights = tf.Variable(next_kernel_data)
                conv = tf.nn.conv2d(ph_input, copy_weights, [1, 1, 1, 1], padding='SAME', name="conv")

                sub_of_kernel_1 = tf.placeholder(tf.float32, shape=[None, None, None, None])
                op_ass_1 = tf.assign(copy_weights, sub_of_kernel_1, validate_shape=False, name="op_ass")

            channel_to_remove = []
            S = []  # contain all the channel
            for i in range(Input_data.shape[3]):  # the channel counts
                if i not in channel_to_remove:
                    S.append(i)  # now, all the channels are here

            next_kernel_selected = next_kernel_data[:, :, channel_to_remove, :]
            Input_selected = Input_data[:, :, :, channel_to_remove]

            step = Input_data.shape[3] * 0.05

            channals_thinet_sorted_index = []

            for count in range(Input_data.shape[3]):

                min = sys.maxsize
                min_index = -1
                for i in S:
                    channel_to_remove.append(i)
                    sess.run(op_ass_1, feed_dict={
                        sub_of_kernel_1: np.insert(next_kernel_selected, channel_to_remove.__len__() - 1,
                                                   next_kernel_data[:, :, i, :], 2)})
                    cur_output = sess.run(conv, feed_dict={
                        ph_input: np.insert(Input_selected, channel_to_remove.__len__() - 1, Input_data[:, :, :, i],
                                            3)})
                    if np.mean(np.square(cur_output)) < min:
                        min = np.mean(np.square(cur_output))
                        min_index = i
                    channel_to_remove.pop()  # pop the element from channel_to_remove list

                channel_to_remove.append(min_index)  # append the channel which need to remove
                S.remove(min_index)  # remove the channel which need to remove in all channel,
                # remain the channel need to save

                next_kernel_selected = np.insert(next_kernel_selected, channel_to_remove.__len__() - 1,
                                                 next_kernel_data[:, :, min_index, :], 2)
                Input_selected = np.insert(Input_selected, channel_to_remove.__len__() - 1,
                                           Input_data[:, :, :, min_index], 3)

                print 'channel_to_remove: ', channel_to_remove
                if count == Input_data.shape[3] - 1:
                    channel_to_remove.reverse()
                    channals_thinet_sorted_index = channel_to_remove
                    print 'channals_thinet_sorted_index: ', channals_thinet_sorted_index

            for ratio in range(1, 20):
                length = int((20 - ratio) * step)
                s = channals_thinet_sorted_index[0:length]
                print 's:', s
                if len(s) >= 1:
                    sess.run(tf.assign(tar_kernel_tensor, tar_kernel_data[:, :, :, s], validate_shape=False))
                    sess.run(tf.assign(tar_bias_tensor, tar_bias_data[s], validate_shape=False))
                    sess.run(tf.assign(tar_mean_tensor, tar_mean_data[s], validate_shape=False))
                    sess.run(tf.assign(tar_var_tensor, tar_var_data[s], validate_shape=False))
                    sess.run(tf.assign(tar_gamma_tensor, tar_gamma_data[s], validate_shape=False))
                    sess.run(tf.assign(tar_beta_tensor, tar_beta_data[s], validate_shape=False))
                    sess.run(tf.assign(next_kernel_tensor, next_kernel_data[:, :, s, :], validate_shape=False))
                    for j in range(0, ds_size, batch_size):
                        sess.run(increment_op)
                    mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable,
                                                         accuracy_summary.mean_variable])
                    sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])
                    f_log.write('rate ' + str(ratio * 0.05) + ',acc ' + str(mean_accuracy) + ',loss ' + str(
                        mean_loss) + '\n')
                    f_log.flush()
                    f_log_S.write('rate ' + str(ratio * 0.05) + ':' + str(s) + '\n')
                    f_log_S.flush()
                    write_2_config_file('Thinet', str(layer), mean_accuracy, str(ratio * 0.05),
                                        config_log_list,
                                        test_acc)
                    f_log_S.write('\n')
                    f_log_S.flush()

        elif layer >= 8 and layer <= 11:
            tar_kernel_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_1/weights:0')
            tar_bias_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_1/biases:0')
            tar_mean_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_1/moving_mean:0')
            tar_var_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_1/moving_variance:0')
            tar_gamma_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_1/gamma:0')
            tar_beta_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_1/beta:0')
            next_bias_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_2/biases:0')
            next_kernel_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_2/weights:0')
            tar_kernel_data, tar_bias_data, tar_mean_data, tar_var_data, tar_gamma_data, tar_beta_data, \
            next_bias_data, next_kernel_data = sess.run([
                tar_kernel_tensor, tar_bias_tensor, tar_mean_tensor, tar_var_tensor, tar_gamma_tensor, tar_beta_tensor, \
                next_bias_tensor, next_kernel_tensor])

            # copy of one conv for greedy iteratio
            with tf.variable_scope('Copy'):
                ph_input = tf.placeholder(tf.float32, shape=[None, None, None, None])
                copy_weights = tf.Variable(next_kernel_data)
                conv = tf.nn.conv2d(ph_input, copy_weights, [1, 1, 1, 1], padding='SAME', name="conv")

                sub_of_kernel_1 = tf.placeholder(tf.float32, shape=[None, None, None, None])
                op_ass_1 = tf.assign(copy_weights, sub_of_kernel_1, validate_shape=False, name="op_ass")

            channel_to_remove = []
            S = []
            for i in range(Input_data.shape[3]):
                if i not in channel_to_remove:
                    S.append(i)
            next_kernel_selected = next_kernel_data[:, :, channel_to_remove, :]
            Input_selected = Input_data[:, :, :, channel_to_remove]

            step = Input_data.shape[3] * 0.05

            channals_thinet_sorted_index = []

            for count in range(Input_data.shape[3]):
                min = sys.maxsize
                min_index = -1
                for i in S:
                    channel_to_remove.append(i)
                    sess.run(op_ass_1, feed_dict={
                        sub_of_kernel_1: np.insert(next_kernel_selected, channel_to_remove.__len__() - 1,
                                                   next_kernel_data[:, :, i, :], 2)})
                    cur_output = sess.run(conv, feed_dict={
                        ph_input: np.insert(Input_selected, channel_to_remove.__len__() - 1, Input_data[:, :, :, i],
                                            3)})
                    if np.mean(np.square(cur_output)) < min:
                        min = np.mean(np.square(cur_output))
                        min_index = i
                    channel_to_remove.pop()

                channel_to_remove.append(min_index)
                S.remove(min_index)
                next_kernel_selected = np.insert(next_kernel_selected, channel_to_remove.__len__() - 1,
                                                 next_kernel_data[:, :, min_index, :], 2)
                Input_selected = np.insert(Input_selected, channel_to_remove.__len__() - 1,
                                           Input_data[:, :, :, min_index], 3)

                print 'channel_to_remove: ', channel_to_remove
                if count == Input_data.shape[3] - 1:
                    channel_to_remove.reverse()
                    channals_thinet_sorted_index = channel_to_remove
                    print 'channals_thinet_sorted_index: ', channals_thinet_sorted_index

            for ratio in range(1, 20):
                length = int((20 - ratio) * step)
                s = channals_thinet_sorted_index[0:length]
                print 's:', s
                if len(s) >= 1:
                    sess.run(tf.assign(tar_kernel_tensor, tar_kernel_data[:, :, :, s], validate_shape=False))
                    sess.run(tf.assign(tar_bias_tensor, tar_bias_data[s], validate_shape=False))
                    sess.run(tf.assign(tar_mean_tensor, tar_mean_data[s], validate_shape=False))
                    sess.run(tf.assign(tar_var_tensor, tar_var_data[s], validate_shape=False))
                    sess.run(tf.assign(tar_gamma_tensor, tar_gamma_data[s], validate_shape=False))
                    sess.run(tf.assign(tar_beta_tensor, tar_beta_data[s], validate_shape=False))
                    sess.run(tf.assign(next_kernel_tensor, next_kernel_data[:, :, s, :], validate_shape=False))
                    for j in range(0, ds_size, batch_size):
                        sess.run(increment_op)
                    mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable,
                                                         accuracy_summary.mean_variable])
                    sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])
                    f_log.write('rate ' + str(0.05 * ratio) + ',acc ' + str(mean_accuracy) + ',loss ' + str(
                        mean_loss) + '\n')
                    f_log.flush()
                    f_log_S.write('rate ' + str(0.05 * ratio) + ':' + str(s) + '\n')
                    f_log_S.flush()
                    write_2_config_file('Thinet', str(layer), mean_accuracy, str(0.05 * ratio), config_log_list,
                                        test_acc)
                    f_log_S.write('\n')
                    f_log_S.flush()
        else:
            print
            "the index is wrong!"


def detect_one_layer_Li(model_class, gen, batch_size, ds_size, load_path, layer, f_log, f_log_S, config_log_list,
                        test_acc):
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NHWC')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss',
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy',
                                       float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op,
                            accuracy_summary.increment_op)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)
        f_log.write('----------- layer ' + str(layer) + '-------------\n')
        f_log_S.write('----------- layer ' + str(layer) + '-------------\n')
        if layer >= 8 and layer <= 11:
            s_kernel_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv/weights:0')
            s_bias_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv/biases:0')
            s_mean_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm/moving_mean:0')
            s_var_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm/moving_variance:0')
            s_gamma_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm/gamma:0')
            s_beta_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm/beta:0')

            _kernel_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_2/weights:0')
            _bias_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_2/biases:0')
            _mean_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_2/moving_mean:0')
            _var_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_2/moving_variance:0')
            _gamma_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_2/gamma:0')
            _beta_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_2/beta:0')

            if layer != 11:
                next_kernel_tensor = get_a_tensor_by_name('Layer' + str(layer + 1) + '/Conv_1/weights:0')
                next_bias_tensor = get_a_tensor_by_name('Layer' + str(layer + 1) + '/Conv_1/biases:0')
                next_s_kernel_tensor = get_a_tensor_by_name('Layer' + str(layer + 1) + '/Conv/weights:0')
                next_s_bias_tensor = get_a_tensor_by_name('Layer' + str(layer + 1) + '/Conv/biases:0')
            else:
                next_kernel_tensor = get_a_tensor_by_name('Layer' + str(layer + 1) + '/Conv/weights:0')
                next_bias_tensor = get_a_tensor_by_name('Layer' + str(layer + 1) + '/Conv/biases:0')

            s_kernel_data, s_bias_data, s_mean_data, s_var_data, s_gamma_data, s_beta_data, \
            _kernel_data, _bias_data, _mean_data, _var_data, _gamma_data, _beta_data, \
            next_kernel_data, next_bias_data = sess.run([
                s_kernel_tensor, s_bias_tensor, s_mean_tensor, s_var_tensor, s_gamma_tensor, s_beta_tensor,
                _kernel_tensor, _bias_tensor, _mean_tensor, _var_tensor, _gamma_tensor, _beta_tensor,
                next_kernel_tensor, next_bias_tensor])
            if layer != 11:
                next_s_kernel_data, next_s_bias_data = sess.run([next_s_kernel_tensor, next_s_bias_tensor])

            shortcut_kernel = s_kernel_data
            shortcut_kernel_abs = np.abs(shortcut_kernel)
            shape = shortcut_kernel.shape
            channal_L1 = []
            for channal in range(shape[3]):
                channal_L1.append(np.sum(shortcut_kernel_abs[:, :, :, channal]))
            channal_L1 = np.array(channal_L1)
            channal_L1_sorted_index = np.argsort(-channal_L1).tolist()
            print 'channal_L1_sorted_index :', channal_L1_sorted_index
            for ratio in range(1, 20):
                length = int((20 - ratio) * 0.05 * shape[3])
                s = channal_L1_sorted_index[0:length]
                print 's:', s
                if len(s) >= 1:
                    sess.run(tf.assign(s_kernel_tensor, s_kernel_data[:, :, :, s], validate_shape=False))
                    sess.run(tf.assign(s_bias_tensor, s_bias_data[s], validate_shape=False))
                    sess.run(tf.assign(s_mean_tensor, s_mean_data[s], validate_shape=False))
                    sess.run(tf.assign(s_var_tensor, s_var_data[s], validate_shape=False))
                    sess.run(tf.assign(s_gamma_tensor, s_gamma_data[s], validate_shape=False))
                    sess.run(tf.assign(s_beta_tensor, s_beta_data[s], validate_shape=False))

                    sess.run(tf.assign(_kernel_tensor, _kernel_data[:, :, :, s], validate_shape=False))
                    sess.run(tf.assign(_bias_tensor, _bias_data[s], validate_shape=False))
                    sess.run(tf.assign(_mean_tensor, _mean_data[s], validate_shape=False))
                    sess.run(tf.assign(_var_tensor, _var_data[s], validate_shape=False))
                    sess.run(tf.assign(_gamma_tensor, _gamma_data[s], validate_shape=False))
                    sess.run(tf.assign(_beta_tensor, _beta_data[s], validate_shape=False))

                    sess.run(tf.assign(next_kernel_tensor, next_kernel_data[:, :, s, :], validate_shape=False))
                    if layer != 11:
                        sess.run(tf.assign(next_s_kernel_tensor, next_s_kernel_data[:, :, s, :], validate_shape=False))

                    for j in range(0, ds_size, batch_size):
                        sess.run(increment_op)
                    mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable,
                                                         accuracy_summary.mean_variable])
                    sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])
                    f_log.write(
                        'rate ' + str(ratio * 0.05) + ',acc ' + str(mean_accuracy) + ',loss ' + str(mean_loss) + '\n')
                    f_log.flush()
                    f_log_S.write('rate ' + str(0.05 * ratio) + ':' + str(s) + '\n')
                    f_log_S.flush()
                    write_2_config_file('L1', str(layer), mean_accuracy, str(ratio * 0.05), config_log_list, test_acc)
        elif layer == 12:
            _kernel_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_1/weights:0')
            _bias_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/Conv_1/biases:0')
            _mean_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_1/moving_mean:0')
            _var_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_1/moving_variance:0')
            _gamma_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_1/gamma:0')
            _beta_tensor = get_a_tensor_by_name('Layer' + str(layer) + '/BatchNorm_1/beta:0')

            fc_weights_tensor = get_a_tensor_by_name('ip/weights:0')
            fc_bias_tensor = get_a_tensor_by_name('ip/biases:0')

            _kernel_data, _bias_data, _mean_data, _var_data, _gamma_data, _beta_data, \
            fc_weights_data, fc_bias_data = sess.run([
                _kernel_tensor, _bias_tensor, _mean_tensor, _var_tensor, _gamma_tensor, _beta_tensor,
                fc_weights_tensor, fc_bias_tensor])

            shortcut_kernel = _kernel_data
            shortcut_kernel_abs = np.abs(shortcut_kernel)
            shape = shortcut_kernel.shape
            channal_L1 = []
            for channal in range(shape[3]):
                channal_L1.append(np.sum(shortcut_kernel_abs[:, :, :, channal]))
            channal_L1 = np.array(channal_L1)
            channal_L1_sorted_index = np.argsort(-channal_L1).tolist()
            print 'channal_L1_sorted_index :', channal_L1_sorted_index
            for ratio in range(1, 20):
                length = int((20 - ratio) * 0.05 * shape[3])  # (20-ratio) * 0.5 * 16
                s = channal_L1_sorted_index[0:length]
                print 's:', s
                if len(s) >= 1:
                    sess.run(tf.assign(_kernel_tensor, _kernel_data[:, :, :, s], validate_shape=False))
                    sess.run(tf.assign(_bias_tensor, _bias_data[s], validate_shape=False))
                    sess.run(tf.assign(_mean_tensor, _mean_data[s], validate_shape=False))
                    sess.run(tf.assign(_var_tensor, _var_data[s], validate_shape=False))
                    sess.run(tf.assign(_gamma_tensor, _gamma_data[s], validate_shape=False))
                    sess.run(tf.assign(_beta_tensor, _beta_data[s], validate_shape=False))
                    sess.run(tf.assign(fc_weights_tensor, fc_weights_data[s, :], validate_shape=False))

                    for j in range(0, ds_size, batch_size):
                        sess.run(increment_op)
                    mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable, \
                                                         accuracy_summary.mean_variable])
                    sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])
                    f_log.write(
                        'rate ' + str(ratio * 0.05) + ',acc ' + str(mean_accuracy) + ',loss ' + str(mean_loss) + '\n')
                    f_log.flush()
                    f_log_S.write('rate ' + str(0.05 * ratio) + ':' + str(s) + '\n')
                    f_log_S.flush()
                    write_2_config_file('L1', str(layer), mean_accuracy, str(ratio * 0.05), config_log_list, test_acc)
        else:
            print
            "wrong layer index!"
