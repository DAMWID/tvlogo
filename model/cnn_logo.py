#!/usr/bin/python

import getopt
import os
import random
import sys
import tensorflow as tf
import numpy as np
from os.path import isdir
from PIL import Image

BATCH_SIZE = 50
TRAIN_EPOCH = 10

KERNEL_SIZE = 5
C1_FEATURES = 32
C2_FEATURES = 64
FC_FEATURES = 1024

label_map = 'label.txt'
train_manifest = 'train.txt'
test_manifest = 'test.txt'
train = None
test = None

restore = False

class Samples(object):
    """Class to maintain sample images and labels"""

    def __init__(self, manifest, label):
        f = open(manifest, 'r')
        self.manifest = [ x.strip() for x in f.readlines() ]
        random.shuffle(self.manifest)
        self.label = label

        self.length = len(self.manifest)
        self.index = 0

        f, _ = self.manifest[0].split(':', 1)
        im = Image.open(f)
        self.width, self.height = im.size
        if im.mode == 'RGB':
            self.channels = 3
        else:
            self.channels = 1

        print('Samples: %d x %d, %d samples' % (self.width, self.height, len(self.manifest)))

    def next_batch(self, batch_size):
        remain = self.length - self.index
        size = min(remain, batch_size)
        xs = np.zeros((size, self.width * self.height * self.channels), dtype=np.float32)
        ys = np.zeros((size, self.label.count), dtype=np.float32)
        for i in range(size):
            f, label = self.manifest[self.index+i].split(':', 1)
            xs[i, :] = np.asarray(Image.open(f), dtype=np.float32).reshape(1, -1)
            ys[i][int(label)] = 1.0
        xs = xs / 255.0
        if remain == 0:
            self.index = 0
        else:
            self.index = self.index + size
        return xs, ys

    def all(self):
        size = len(self.manifest)
        xs = np.zeros((size, self.width * self.height * self.channels), dtype=np.float32)
        ys = np.zeros((size, self.label.count), dtype=np.float32)
        for i in range(self.length):
            f, label = self.manifest[i].split(':', 1)
            xs[i, :] = np.ravel(np.asarray(Image.open(f), dtype=np.float32))
            ys[i][int(label)] = 1.0
        xs = xs / 255.0
        return xs, ys

class Labels:
    """Class to maintain label definitions"""

    def __init__(self, mapfile):
        with open(mapfile) as f:
            content = [ x.strip() for x in f.readlines() ]
            self.count = len(content)
            self.map = {}
            for line in content:
                idx, ch = line.split(':', 1)
                self.map[idx] = ch

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hb:e:k:l:rt:v:", ['help', 'batch=', 'epoch=', 'kernel=', 'label=', 'restore', 'train=', 'verify='])
    except getopt.GetoptError, msg:
        raise Usage(msg)

    if len(args) != 1 or not isdir(args[0]):
        raise Usage(None)
    basedir = args[0]

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            raise Usage(None)
        elif opt in ('-b', '--batch'):
            BATCH_SIZE = int(arg)
        elif opt in ('-e', '--epoch'):
            TRAIN_EPOCH = int(arg)
        elif opt in ('-k', '--kernel'):
            KERNEL_SIZE = int(arg)
        elif opt in ('-l', '--label'):
            label_map = arg
        elif opt in ('-r', '--restore'):
            restore = True
        elif opt in ('-t', '--train'):
            train_manifest = arg
        elif opt in ('-v', '--verify'):
            test_manifest = arg
except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s OPTION [dir]' % sys.argv[0])
    sys.exit(2)

print("[ kernel_size=%d, epoch=%d, batch_size=%d ]" % (KERNEL_SIZE, TRAIN_EPOCH, BATCH_SIZE))

os.chdir(basedir)

label = Labels(label_map)
train = Samples(train_manifest, label)
test = Samples(test_manifest, label)

width, height, channels = train.width, train.height, train.channels

features = width * height * channels

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, features])
y_ = tf.placeholder(tf.float32, shape=[None, label.count])

# Convolutional Neuron Network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, channels, C1_FEATURES])
b_conv1 = bias_variable([C1_FEATURES])

x_image = tf.reshape(x, [-1, height, width, channels])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, C1_FEATURES, C2_FEATURES])
b_conv2 = bias_variable([C2_FEATURES])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([(height/4) * (width/4) * C2_FEATURES, FC_FEATURES])
b_fc1 = bias_variable([FC_FEATURES])

h_pool2_flat = tf.reshape(h_pool2, [-1, (height/4)*(width/4)*C2_FEATURES])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([FC_FEATURES, label.count])
b_fc2 = bias_variable([label.count])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

checkpoint = "model-%d_%d_%d_%d_%d" % (KERNEL_SIZE, KERNEL_SIZE, C1_FEATURES, C2_FEATURES, FC_FEATURES)

if restore:
    saver.restore(sess, checkpoint)
    print('Model restored from %s' % checkpoint)
else:
    for e in range(TRAIN_EPOCH):
        b = 0
        while True:
            batch = train.next_batch(BATCH_SIZE)
            if batch[0].shape[0] == 0:
                break
            if b % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("Epoch %d, batch %d, training accuracy %g" % (e, b, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            b += 1

    saver.save(sess, checkpoint)
    print('Model saved to %s' % checkpoint)

failed = open('cnn_test_failed.txt', 'w')
n_failed = 0
total = 0
b = 0
while True:
    batch = test.next_batch(BATCH_SIZE)
    num = batch[0].shape[0]
    if num == 0:
        break
    p = correct_prediction.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    train_accuracy = np.mean(p.astype('float32'))
    print("Batch %d, testing accuracy %g" % (b, train_accuracy))
    for i, correct in enumerate(p):
        if correct:
            continue

        yt = y_conv.eval(feed_dict={x: batch[0][i:i+1, :], y_: batch[1][i:i+1, :], keep_prob: 1.0})
        n_failed += 1
        f, _ = test.manifest[total+i].split(':', 1)
        failed.write("%s: [ " % f)
        for j in range(yt.shape[1]):
            failed.write("%0.4f, " % yt[0][j])
        failed.write("] \n")
    total += num
    b += 1

print("test accuracy %g" % (float(total-n_failed)/float(total)))
