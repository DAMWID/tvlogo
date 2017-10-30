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

VGG16_LAYOUT = ((224, 224, 3), ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)), (4096, 4096), 0)
CUSTOM_LAYOUT = ((224, 224, 3), ((32, 32), (64, 64), (128, 128)), (1024,), 0)
SIMPLE_LAYOUT = ((224, 224, 3), ((32, 32), (64, 64)), (1024,), 0)
MINIMAL_LAYOUT = ((224, 224, 3), ((32, 32), (64,)), (1024,), 0)
LINEAR_LAYOUT = ((224, 224, 3), (), (), 0)

label_manifest = 'channels.map'

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
        bx = (bx - 128.0) / 128.0
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
        bx = (bx - 128.0) / 128.0
        return bx, by, bf

class Labels(object):
    """Class to maintain label definitions"""

    def __init__(self, labelfile):
        with open(labelfile) as f:
            contents = [ l.strip() for l in f.readlines() if not l.startswith('#') ]
            self.count = [ int(l.split(':', 2)[0]) for l in contents if l.split(':', 2)[1] == 'max' ][0]
            self.db = [('_', 'unknown')] * self.count
            for l in contents:
                i, ch, name = l.split(':', 2)
                if ch.isdigit():
                    self.db[int(i)] = (ch, name)

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hb:d:e:l:n:r", ['help', 'batch=', 'dir=', 'epoch=', 'label=', 'network=', 'restart'])
    except getopt.GetoptError, msg:
        raise Usage(msg)

    if len(args) != 2 or args[0] not in ('train', 'validate') or not isfile(args[1]):
        raise Usage(None)
    action, manifest = args

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
            label_manifest = arg
        elif opt in ('-n', '--network'):
            if arg == 'vgg16':
                layout = list(VGG16_LAYOUT)
            elif arg == 'custom':
                layout = list(CUSTOM_LAYOUT)
            elif arg == 'simple':
                layout = list(SIMPLE_LAYOUT)
            elif arg == 'minimal':
                layout = list(MINIMAL_LAYOUT)
            elif arg == 'linear':
                layout = list(LINEAR_LAYOUT)
        elif opt in ('-r', '--restart'):
            restart = True

    if not isdir(basedir):
        raise Usage('Directory not found: %s' % basedir)
except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s OPTION <train|validate> manifest_file' % sys.argv[0])
    sys.exit(2)

os.chdir(basedir)

labels = Labels(label_manifest)
samples = Samples(manifest, labels.count)

layout[0] = (samples.height, samples.width, samples.channels)
layout[-1] = labels.count

attr_i = 'x'.join([ str(l) for l in layout[0] ])
attr_c = '_'.join([ 'x'.join([ str(c) for c in l ]) for l in layout[1] ])
attr_f = '_'.join([ str(l) for l in layout[2] ])
attr = '-'.join([ s for s in (attr_i, attr_c, attr_f, str(layout[-1])) if len(s) > 0 ])

model_param_path = join(basedir, 'pretrained', 'tvlogo-vgg-%s.npy' % attr)
if restart or not isfile(model_param_path):
    try:
        os.makedirs(dirname(model_param_path))
    except OSError:
        pass
    model = vgg.VGG(layout)
else:
    model = vgg.VGG(layout, model_param_path)

if action == 'validate':
    epoch = 1
    failed = open('failed.txt', 'w')
    n_failed = 0

remaining = samples.count * epoch
batches = (samples.count + batch_size - 1) / batch_size
for e in range(epoch):
    b = 0
    processed = 0
    t_start = time.time()
    while True:
        batch = samples.next_batch(batch_size)
        num = batch[0].shape[0]
        if num == 0:
            break
        if action == 'train':
            model.train(batch[0], batch[1])
        else:
            prob, loss = model.validate(batch[0], batch[1])
            f_idx = np.flatnonzero(np.not_equal(np.argmax(prob, 1), np.argmax(batch[1], 1)))
            for i in f_idx:
                failed.write("%s: [ " % batch[2][i])
                for j in np.argsort(prob[i])[:-6:-1]:
                    failed.write("%s (%0.4f), " % (labels.db[j][0], prob[i][j]))
                failed.write("] \n")
            n_failed += len(f_idx)
        remaining -= num
        processed += num
        if b % 10 == 0:
            if action == 'train':
                prob, loss = model.validate(batch[0], batch[1])
            accuracy = np.mean(np.equal(np.argmax(prob, 1), np.argmax(batch[1], 1)).astype('float32'))
            t = time.time()
            dt, t_start = t - t_start, t
            r_sec = int(remaining * dt / processed)
            r_hour, r_min, r_sec = r_sec / 3600, (r_sec % 3600) / 60, r_sec % 60
            print("%s: epoch %d/%d, batch %d/%d, accuracy: %6.2f%%, loss: %6.4f, time remaining: %02d:%02d:%02d (%3.1f exapmles/sec)" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), e+1, epoch, b, batches, accuracy*100, loss, r_hour, r_min, r_sec, processed/dt))
            processed = 0
        b += 1

if action == 'train':
    model.save(model_param_path)
else:
    print("Overall validate accuracy %6.2f%%" % ((samples.count-n_failed)*100.0/samples.count))
