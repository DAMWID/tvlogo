#!/usr/bin/python

import glob
import os
import random
import sys
from os import listdir
from os.path import isdir, join, expanduser, relpath, basename, dirname

if len(sys.argv) < 2 or not isdir(sys.argv[1]):
    print('Usage: %s <DIR>' % sys.argv[0])
    sys.exit()

basedir = sys.argv[1]
logodir = join(basedir, 'logo')

ch_index = {}
with open(join(basedir, 'channels.txt')) as f:
    contents = [ l.strip() for l in f.readlines() if not l.startswith('#') ]
    channels = [ int(l.split(':', 1)[0]) for l in contents ]
    for i, ch in enumerate(channels):
        ch_index[ch] = i

logofiles = [ relpath(f, basedir) for f in glob.glob(join(logodir, '*', 'logo-*.jpg')) ]
random.shuffle(logofiles)
testfiles = logofiles[:len(logofiles)/4]
trainfiles = logofiles[len(logofiles)/4:]

with open(join(basedir, 'all.txt'), 'w') as test:
    for f in logofiles:
        ch = int(basename(dirname(f)))
        if ch in ch_index:
            test.write('%s:%d\n' % (f, ch_index[ch]))

with open(join(basedir, 'test.txt'), 'w') as test:
    for f in testfiles:
        ch = int(basename(dirname(f)))
        if ch in ch_index:
            test.write('%s:%d\n' % (f, ch_index[ch]))

with open(join(basedir, 'train.txt'), 'w') as train:
    for f in trainfiles:
        ch = int(basename(dirname(f)))
        if ch in ch_index:
            train.write('%s:%d\n' % (f, ch_index[ch]))
