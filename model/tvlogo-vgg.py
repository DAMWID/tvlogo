#!/usr/bin/python

import getopt
import os
import random
import sys
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from os.path import isdir, isfile, join, basename, dirname
from PIL import Image
import vgg

DEFAULT_BATCH_SIZE = 50
DEFAULT_EPOCH = 5
MAX_LABELS = 1000

VGG16_LAYOUT = ((224, 224, 3), ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)), (4096, 4096), MAX_LABELS)
CUSTOM_LAYOUT = ((224, 224, 3), ((32, 32), (64, 64), (128, 128)), (1024,), MAX_LABELS)
SIMPLE_LAYOUT = ((224, 224, 3), ((32, 32), (64, 64)), (1024,), 0)
LINEAR_LAYOUT = ((224, 224, 3), (), (), 0)

label_manifest = 'channels.txt'
train_manifest = 'train.txt'
eval_manifest = 'evaluate.txt'

basedir = '.'
restart = False
epoch = DEFAULT_EPOCH
batch_size = DEFAULT_BATCH_SIZE
layout = list(CUSTOM_LAYOUT)

class Samples(object):
    """Class to maintain sample images and labels"""

    def __init__(self, manifest, labels):
        f = open(manifest, 'r')
        self.manifest = [ x.strip() for x in f.readlines() ]
        random.shuffle(self.manifest)
        self.labels = labels

        self.count = len(self.manifest)
        self.index = 0

        f, _ = self.manifest[0].split(':', 1)
        im = Image.open(f)
        self.width, self.height = im.size
        if im.mode == 'RGB':
            self.channels = 3
        else:
            self.channels = 1

        print('Samples: %d x %d, %d samples' % (self.width, self.height, len(self.manifest)))

    def next_batch(self, size):
        remain = self.count - self.index
        size = min(remain, size)
        bx = np.zeros((size, self.width * self.height * self.channels), dtype=np.float32)
        by = np.zeros((size, self.labels), dtype=np.float32)
        bf = []
        for i in range(size):
            f, label = self.manifest[self.index+i].split(':', 1)
            bx[i, :] = np.asarray(Image.open(f), dtype=np.float32).reshape(1, -1)
            by[i][int(label)] = 1.0
            bf.append(f)
        bx = bx / 255.0
        if remain == 0:
            self.index = 0
        else:
            self.index = self.index + size
        return bx, by, bf

    def all(self):
        size = len(self.manifest)
        bx = np.zeros((size, self.width * self.height * self.channels), dtype=np.float32)
        by = np.zeros((size, self.labels), dtype=np.float32)
        bf = []
        for i in range(self.count):
            f, label = self.manifest[i].split(':', 1)
            bx[i, :] = np.ravel(np.asarray(Image.open(f), dtype=np.float32))
            by[i][int(label)] = 1.0
            bf.append(f)
        bx = bx / 255.0
        return bx, by, bf

class Labels(object):
    """Class to maintain label definitions"""

    def __init__(self, labelfile, labels):
        with open(labelfile) as f:
            contents = [ l.strip() for l in f.readlines() if not l.startswith('#') ]
            self.count = labels if labels > len(contents) else len(contents)
            self.db = [('_', 'Unknow')] * self.count
            for i, l in enumerate(contents):
                self.db[i] = l.split(':', 1)

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hb:d:e:l:n:rt:v:", ['help', 'batch=', 'dir=', 'epoch=', 'label=', 'network=', 'restart', 'train=', 'evaluate='])
    except getopt.GetoptError, msg:
        raise Usage(msg)

    if len(args) != 1:
        raise Usage(None)
    action = args[0]

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            raise Usage(None)
        elif opt in ('-b', '--batch'):
            batch_size = int(arg)
        elif opt in ('-d', '--dir'):
            basedir = arg
        elif opt in ('-e', '--epoch'):
            epoch = int(arg)
        elif opt in ('-l', '--label'):
            label_map = arg
        elif opt in ('-n', '--network'):
            if arg == 'vgg16':
                layout = list(VGG16_LAYOUT)
            elif arg == 'custom':
                layout = list(CUSTOM_LAYOUT)
            elif arg == 'simple':
                layout = list(SIMPLE_LAYOUT)
            elif arg == 'linear':
                layout = list(LINEAR_LAYOUT)
        elif opt in ('-r', '--restart'):
            restart = True
        elif opt in ('-t', '--train'):
            train_manifest = arg
        elif opt in ('-v', '--evaluate'):
            eval_manifest = arg

    if not isdir(basedir):
        raise Usage('Directory not found: %s' % basedir)
except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s OPTION <train|evaluate>' % sys.argv[0])
    sys.exit(2)

print("[ epoch=%d, batch_size=%d ]" % (epoch, batch_size))

os.chdir(basedir)

label = Labels(label_manifest, layout[-1])
train = Samples(train_manifest, label.count)
evaluate = Samples(eval_manifest, label.count)

layout[0] = (train.height, train.width, train.channels)
layout[-1] = label.count

attr_i = 'x'.join([ str(l) for l in layout[0] ])
attr_c = '_'.join([ str(l[0]) for l in layout[1] ])
attr_f = '_'.join([ str(l) for l in layout[2] ])
attr = '-'.join([ s for s in (attr_i, attr_c, attr_f, str(layout[-1])) if len(s) > 0 ])

model_param_path = join(basedir, 'pretrained', 'tvlogo-vgg-%s.npy' % attr)
try:
    os.makedirs(dirname(model_param_path))
except OSError:
    pass

if restart or not isfile(model_param_path):
    model = vgg.VGG(layout)
else:
    model = vgg.VGG(layout, model_param_path)

if action == 'train':
    remaining = train.count * epoch
    batches = (train.count + batch_size - 1) / batch_size
    for e in range(epoch):
        b = 0
        processed = 0
        t_start = time.time()
        while True:
            batch = train.next_batch(batch_size)
            num = batch[0].shape[0]
            if num == 0:
                break
            model.train(batch[0], batch[1])
            remaining -= num
            processed += num
            if b % 10 == 0:
                prob, loss = model.evaluate(batch[0], batch[1])
                accuracy = np.mean(np.equal(np.argmax(prob, 1), np.argmax(batch[1], 1)).astype('float32'))
                t = time.time()
                dt, t_start = t - t_start, t
                r_sec = int(remaining * dt / processed)
                r_hour, r_min, r_sec = r_sec / 3600, (r_sec % 3600) / 60, r_sec % 60
                print("%s: epoch %d/%d, batch %d/%d, accuracy: %6.2f%%, loss: %6.4f, time remaining: %02d:%02d:%02d (%3.1f exapmles/sec)" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), e+1, epoch, b, batches, accuracy*100, loss, r_hour, r_min, r_sec, processed/dt))
                processed = 0
            b += 1
    model.save(model_param_path)

failed = open('failed.txt', 'w')
n_failed = 0
b = 0
remaining = evaluate.count
batches = (evaluate.count + batch_size - 1) / batch_size
processed = 0
t_start = time.time()
while True:
    batch = evaluate.next_batch(batch_size)
    num = batch[0].shape[0]
    if num == 0:
        break
    prob, loss = model.evaluate(batch[0], batch[1])
    p = np.not_equal(np.argmax(prob, 1), np.argmax(batch[1], 1))
    accuracy = 1.0 - np.mean(p.astype('float32'))
    f_idx = np.flatnonzero(p)
    for i in f_idx:
        failed.write("%s: [ " % batch[2][i])
        for j in np.argsort(prob[i])[:-6:-1]:
            failed.write("%s (%0.4f), " % (label.db[j][0], prob[i][j]))
        failed.write("] \n")
    n_failed += len(f_idx)
    remaining -= num
    processed += num
    if b % 10 == 0:
        t = time.time()
        dt, t_start = t - t_start, t
        r_sec = int(remaining * dt / processed)
        r_hour, r_min, r_sec = r_sec / 3600, (r_sec % 3600) / 60, r_sec % 60
        print("%s: batch %d/%d, accuracy: %6.2f%%, loss: %6.4f, time remaining: %02d:%02d:%02d (%3.1f exapmles/sec)" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), b, batches, accuracy*100, loss, r_hour, r_min, r_sec, processed/dt))
        processed = 0
    b += 1

print("Overall evaluate accuracy %5.1f%%" % ((evaluate.count-n_failed)*100.0/evaluate.count))
