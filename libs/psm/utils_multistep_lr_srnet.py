import tensorflow as tf
import time
from queues import *
from generator_psm import *
import pandas as pd


class average_summary:
    def __init__(self, variable, name, num_iterations):
        self.sum_variable = tf.get_variable(name, shape=[],
                                            initializer=tf.constant_initializer(0),
                                            dtype=variable.dtype.base_dtype,
                                            trainable=False,
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        with tf.control_dependencies([variable]):
            self.increment_op = tf.assign_add(self.sum_variable, variable)
        self.mean_variable = self.sum_variable / float(num_iterations)
        self.summary = tf.summary.scalar(name, self.mean_variable)
        with tf.control_dependencies([self.summary]):
            self.reset_variable_op = tf.assign(self.sum_variable, 0)

    def add_summary(self, sess, writer, step):
        s, _ = sess.run([self.summary, self.reset_variable_op])
        writer.add_summary(s, step)


class Model:
    def __init__(self, is_training=None, data_format='NCHW'):
        self.data_format = data_format
        if is_training is None:
            self.is_training = tf.get_variable('is_training', dtype=tf.bool,
                                               initializer=tf.constant_initializer(True),
                                               trainable=False)
        else:
            self.is_training = is_training

    def _build_model(self, inputs):
        raise NotImplementedError('Here is your model definition')

    def _build_losses(self, labels):
        self.labels = tf.cast(labels, tf.int64)
        with tf.variable_scope('loss'):
            oh = tf.one_hot(self.labels, 2)
            xen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=oh, logits=self.outputs))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.add_n([xen_loss] + reg_losses)
        with tf.variable_scope('accuracy'):
            am = tf.argmax(self.outputs, 1)
            equal = tf.equal(am, self.labels)  # am:predict
            self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
        return self.loss, self.accuracy

    def _build_predict(self):
        with tf.variable_scope('prob'):
            self.test_result = tf.nn.softmax(self.outputs)
        return self.test_result


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


def train(model_class, train_gen, valid_gen, train_batch_size,
          valid_batch_size, valid_ds_size, test_gen, test_batch_size, test_interval, test_ds_size, optimizer,
          boundaries, values,
          train_interval, valid_interval, max_iter,
          save_interval, log_path, num_runner_threads=1,
          load_path=None):
    tf.reset_default_graph()
    train_runner = GeneratorRunner(train_gen, train_batch_size * 10)
    valid_runner = GeneratorRunner(valid_gen, valid_batch_size * 10)
    test_runner = GeneratorRunner(test_gen, test_batch_size * 10)

    is_training = tf.get_variable('is_training', dtype=tf.bool,
                                  initializer=True, trainable=False)
    data_sel = tf.get_variable('data_sel', dtype=tf.int32,
                               initializer=0, trainable=False)
    # data_sel = 0 for train, 1 for valid, 2 for test

    if train_batch_size == valid_batch_size:
        batch_size = train_batch_size
        disable_training_op = tf.assign(is_training, False)
        enable_training_op = tf.assign(is_training, True)
    else:
        batch_size = tf.get_variable('batch_size', dtype=tf.int32,
                                     initializer=train_batch_size,
                                     trainable=False,
                                     collections=[tf.GraphKeys.LOCAL_VARIABLES])
        disable_training_op = tf.group(tf.assign(is_training, False),
                                       tf.assign(batch_size, valid_batch_size),
                                       tf.assign(data_sel, 1))
        enable_test_op = tf.group(tf.assign(is_training, False), tf.assign(batch_size, test_batch_size),
                                  tf.assign(data_sel, 2))
        enable_training_op = tf.group(tf.assign(is_training, True),
                                      tf.assign(batch_size, train_batch_size),
                                      tf.assign(data_sel, 0))

    img_batch, label_batch = queueSelection([train_runner, valid_runner, test_runner],
                                            tf.cast(data_sel, tf.int32),
                                            batch_size)
    model = model_class(is_training, 'NCHW')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    train_loss_s = average_summary(loss, 'train_loss', train_interval)
    train_accuracy_s = average_summary(accuracy, 'train_accuracy',
                                       train_interval)
    valid_loss_s = average_summary(loss, 'valid_loss',
                                   float(valid_ds_size) / float(valid_batch_size))
    valid_accuracy_s = average_summary(accuracy, 'valid_accuracy',
                                       float(valid_ds_size) / float(valid_batch_size))

    test_loss_s = average_summary(loss, 'test_loss',
                                  float(test_ds_size) / float(test_batch_size))
    test_accuracy_s = average_summary(accuracy, 'test_accuracy',
                                      float(test_ds_size) / float(test_batch_size))

    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)
    optimizer = optimizer(learning_rate)

    minimize_op = optimizer.minimize(loss, global_step)
    train_op = tf.group(minimize_op, train_loss_s.increment_op,
                        train_accuracy_s.increment_op)
    increment_valid = tf.group(valid_loss_s.increment_op,
                               valid_accuracy_s.increment_op)

    increment_test = tf.group(test_loss_s.increment_op,
                              test_accuracy_s.increment_op)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # The following code calculates the parameters for each convolutional layer and the total parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()

        # condition == 4, Focus only on convolutional layers; 
        # condition > 0, Focus on all layers;
        if len(shape) == 4:          
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print 'variable_parameters:', variable_parameters
            total_parameters += variable_parameters
    print "total_parameters:", total_parameters

    # for all data collection(gpu memory etc.)
    run_metadata = tf.RunMetadata()

    # varlist = optimistic_restore_vars(load_path)
    # saver = tf.train.Saver(varlist, max_to_keep=10000)
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        # sess.run(init_op)
        
        # Write down all the data by using RunMetadata
        sess.run(init_op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, 
                                                output_partition_graphs=True), run_metadata=run_metadata)
        # write it to txt file
        with open("calpa_srnet_init_full_traces.txt", "a") as out: 
            out.write(str(run_metadata))
            print "log_gpu_memory_finished...."

        if load_path is not None:
            saver.restore(sess, load_path)
        train_runner.start_threads(sess, num_runner_threads)
        valid_runner.start_threads(sess, 1)
        test_runner.start_threads(sess, 1)

        writer = tf.summary.FileWriter(log_path + '/LogFile/',
                                       sess.graph)
        start = sess.run(global_step)
        sess.run([train_loss_s.reset_variable_op,
                  train_accuracy_s.reset_variable_op,
                  valid_loss_s.reset_variable_op,
                  valid_accuracy_s.reset_variable_op,
                  test_loss_s.reset_variable_op,
                  test_accuracy_s.reset_variable_op])

        sess.run(enable_training_op)
        print "network will be evaluatd every %i iterations on validation set" % valid_interval
        for i in xrange(start + 1, max_iter + 1):
            sess.run(train_op)
            if i % train_interval == 0:
                train_loss_s.add_summary(sess, writer, i)
                train_accuracy_s.add_summary(sess, writer, i)
                s = sess.run(lr_summary)
                writer.add_summary(s, i)
            if i % valid_interval == 0:
                sess.run(disable_training_op)
                print 'valid: batch_size:', sess.run(batch_size)
                _time = time.time()
                for j in range(0, valid_ds_size, valid_batch_size):
                    sess.run([increment_valid])
                _acc_val = sess.run(valid_accuracy_s.mean_variable)
                _loss_val = sess.run(valid_loss_s.mean_variable)
                print 'iter:', i, "-- accuracy on validation set:", _acc_val, "-- loss:", _loss_val
                print "evaluation time on validation set:", time.time() - _time, "seconds"
                valid_loss_s.add_summary(sess, writer, i)
                valid_accuracy_s.add_summary(sess, writer, i)
                sess.run(enable_training_op)
            if i % test_interval == 0:
                sess.run(enable_test_op)
                print 'test: batch_size:', sess.run(batch_size)
                _time = time.time()
                for j in range(0, test_ds_size, test_batch_size):
                    sess.run([increment_test])
                _acc_val = sess.run(test_accuracy_s.mean_variable)
                _loss_val = sess.run(test_loss_s.mean_variable)
                print 'iter:', i, "-- accuracy on test set:", _acc_val, "-- loss:", _loss_val
                print "test time on test set:", time.time() - _time, "seconds"
                test_loss_s.add_summary(sess, writer, i)
                test_accuracy_s.add_summary(sess, writer, i)
                sess.run(enable_training_op)
            if i % save_interval == 0:
                saver.save(sess, log_path + '/Model_' + str(i) + '.ckpt')


def test_dataset_and_get_csv(model_class, gen, batch_size, ds_size, load_path, csv_path):
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NCHW')
    model._build_model(img_batch)

    # valid stego's confidence
    valid_pred = model._build_predict()
    # valid stego's label [0,1]
    label_output = tf.one_hot(tf.cast(label_batch, tf.int64), 2)

    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss',
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy',
                                       float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op,
                            accuracy_summary.increment_op)
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)

    with tf.Session() as sess:
        predict_list = []
        stego_label_list = []

        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)
        for j in range(0, ds_size, batch_size):
            sess.run(increment_op)

            valid_label = sess.run(label_output)
            valid_confi = sess.run(valid_pred)
            for k in range(len(valid_confi)):
                predict_list.append(valid_confi[k][1])
            for k in range(len(valid_label)):
                stego_label_list.append(int(valid_label[k][1]))

        # write to csv
        df = pd.DataFrame({'Predict': predict_list, 'Label': stego_label_list})
        df.to_csv(csv_path, mode='w', index=True, sep=',', columns=['Predict', 'Label'],
                  header=0)

        mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable,
                                             accuracy_summary.mean_variable])
    print "Accuracy:", mean_accuracy, " | Loss:", mean_loss


def test_dataset(model_class, gen, batch_size, ds_size, load_path):
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NCHW')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss',
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy',
                                       float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op,
                            accuracy_summary.increment_op)
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)
        for j in range(0, ds_size, batch_size):
            sess.run(increment_op)
        test_mean_loss, test_mean_accuracy = sess.run([loss_summary.mean_variable,
                                                       accuracy_summary.mean_variable])
    print "Accuracy:", test_mean_accuracy, " | Loss:", test_mean_loss


### Implementation of Adamax optimizer, taken from : https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class AdamaxOptimizer(optimizer.Optimizer):
    """
    Optimizer that implements the Adamax algorithm. 
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
