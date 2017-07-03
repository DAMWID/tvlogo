import tensorflow as tf

import numpy as np
from functools import reduce

class VGG(object):
    """
    A custom VGG model for image classification.
    """

    def __init__(self, layout, param_path=None, dropout=0.5):
        if param_path is not None:
            self.data_dict = np.load(param_path, encoding='latin1').item()
            print("Model parameters loaded from %s" % param_path)
        else:
            self.data_dict = None

        self.var_dict = {}
        self.train_dropout = dropout

        self.sess = tf.InteractiveSession()

        height, width, depth = layout[0];
        self.x = tf.placeholder(tf.float32, shape=[None, height * width * depth])
        self.labels = tf.placeholder(tf.float32, shape=[None, layout[-1]])
        self.dropout = tf.placeholder(tf.float32)

        self.images = tf.reshape(self.x, [-1, height, width, depth])

        bottom_layer = (self.images, height, width, depth)
        for i, layer in enumerate(layout[1]):
            for j, channels in enumerate(layer):
                conv = self.conv_layer(bottom_layer[0], bottom_layer[3], channels, "conv%d_%d" % (i+1, j+1))
                bottom_layer = (conv, bottom_layer[1], bottom_layer[2], channels)
            pool = self.max_pool(bottom_layer[0], 'pool%d' % (i+1))
            bottom_layer = (pool, bottom_layer[1]/2, bottom_layer[2]/2, bottom_layer[3])

        bottom_layer = (bottom_layer[0], bottom_layer[1]*bottom_layer[2]*bottom_layer[3])
        for i, channels in enumerate(layout[2]):
            fc = self.fc_layer(bottom_layer[0], bottom_layer[1], channels, "fc%d" % (i+1))
            relu = tf.nn.relu(fc)
            drop = tf.nn.dropout(relu, self.dropout)
            bottom_layer = (drop, channels)

        self.y = self.fc_layer(bottom_layer[0], bottom_layer[1], layout[-1], "y")

        self.prob = tf.nn.softmax(self.y, name="prob")

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.y))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        self.sess.run(tf.global_variables_initializer())

        self.data_dict = None
        print("Model initialized")

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        print("CONV layer %s: %d x %d" % (name, in_channels, out_channels))
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        print("FC layer %s: %d x %d" % (name, in_size, out_size))
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.1)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], stddev=0.1)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], stddev=0.1)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], stddev=0.1)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        var = tf.Variable(value, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save(self, param_path):
        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = self.sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(param_path, data_dict)
        print(("Model parameters saved to %s" % param_path))
        return param_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def train(self, x, y):
        self.sess.run(self.train_step, feed_dict={self.x: x, self.labels: y, self.dropout: self.train_dropout})

    def evaluate(self, x, y):
        return self.sess.run([self.prob, self.cross_entropy], feed_dict={self.x: x, self.labels: y, self.dropout: 1.0})

    def infer(self, x):
        return self.sess.run(self.prob, feed_dict={self.x: x, self.dropout: 1.0})
