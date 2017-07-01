#!/usr/bin/python

import os
import random
import sys
import tensorflow as tf
import numpy as np
from PIL import Image

class tvlogo(object):
    """Class to maintain tv logo data and label"""

    def __init__(self, manifest, channels):
        f = open(manifest, 'r')
        self.manifest = [ x.strip() for x in f.readlines() ]
        random.shuffle(self.manifest)
        self.channels = channels

        self.length = len(self.manifest)
        self.index = 0

        f, _ = self.manifest[0].split(':', 1)
        im = Image.open(f)
        self.width, self.height = im.size

        print('tvlogo: %d x %d, %d samples' % (self.width, self.height, len(self.manifest)))

    def next_batch(self, batch):
        xs = np.zeros((batch, self.width * self.height * 3), dtype=np.float32)
        ys = np.zeros((batch, self.channels), dtype=np.float32)
        for i in range(batch):
            idx = (self.index + i) % self.length
            f, label = self.manifest[idx].split(':', 1)
            xs[i, :] = np.asarray(Image.open(f), dtype=np.float32).reshape(1, -1)
            ys[i][int(label)] = 1.0
        self.index = (self.index + batch) % self.length
        xs = xs / 255.0
        return xs, ys

    def all(self):
        batch = len(self.manifest)
        xs = np.zeros((batch, self.width * self.height * 3), dtype=np.float32)
        ys = np.zeros((batch, self.channels), dtype=np.float32)
        for i in range(self.length):
            f, label = self.manifest[i].split(':', 1)
            xs[i, :] = np.ravel(np.asarray(Image.open(f), dtype=np.float32))
            ys[i][int(label)] = 1.0
        xs = xs / 255.0
        return xs, ys

f = open('channel.map', 'r')
content = [ x.strip() for x in f.readlines() ]
channels = len(content)
channel_map = {}
for line in content:
    label, ch = line.split(':', 1)
    channel_map[label] = ch

train = tvlogo('train.txt', channels)
test = tvlogo('test.txt', channels)

features = train.width * train.height * 3

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, channels])

W = tf.Variable(tf.zeros([features, channels]))
b = tf.Variable(tf.zeros([channels]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for n in range(1, 10001):
    batch = train.next_batch(50)
    if n > 0 and n % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print("Step %d, training accuracy %g" % (n, train_accuracy))
    sess.run(train_step, feed_dict = {x:batch[0], y_:batch[1]})

batch = test.all()
#print(sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]}))
print("test accuracy %g" % accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))
w = np.transpose(W.eval())
min = w.min()
max = w.max()
w = (255 * (w - min) / (max - min)).astype('uint8')

# convert weight to image
col = 10
row = (channels + col - 1) / col
img_a = np.zeros((train.height * row, train.width * col, 3), dtype=np.uint8)
for i in range(channels):
    r = i / col
    c = i % col
    img_a[train.height*r:train.height*(r+1), train.width*c:train.width*(c+1), :] = w[i, :].reshape(-1, train.width, 3)

Image.fromarray(img_a).save('weight.png')

batch = test.all()
yt = y.eval(feed_dict={x: batch[0], y_: batch[1]})
p = correct_prediction.eval(feed_dict={x: batch[0], y_: batch[1]})
num = yt.shape[0]
error = open('failed.txt', 'w')
for i in range(num):
    if p[i] == True:
        continue

    f, _ = test.manifest[i].split(':', 1)
    error.write("%s: [ " % f)
    for j in range(channels):
        error.write("%0.4f, " % yt[i][j])
    error.write("] \n")

