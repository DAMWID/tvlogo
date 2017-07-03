#!/usr/bin/python

import getopt
import glob
import os
import random
import sys
from os import listdir
from os.path import isdir, join, expanduser, relpath, basename, dirname

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

evaldir = None
traindir = None
orig_only = False

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "he:ot:", ['help', 'evaluate=', 'original', 'train='])
    except getopt.GetoptError, msg:
        raise Usage(msg)

    if len(args) != 1 or not isdir(args[0]):
        raise Usage(None)
    basedir = args[0]

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            raise Usage(None)
        elif opt in ('-e', '--evaluate'):
            evaldir = arg
        elif opt in ('-o', '--original'):
            orig_only = True
        elif opt in ('-t', '--train'):
            traindir = arg

    if evaldir and not isdir(evaldir):
        raise Usage('Directory %s not exists' % evaldir)
    if traindir and not isdir(traindir):
        raise Usage('Directory %s not exists' % traindir)
    if not evaldir and not traindir:
        raise Usage('No source directory defined')
except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s [-e evaluate_path ] [-t train_path ] <target_path>' % sys.argv[0])
    sys.exit(2)

ch_index = {}
with open(join(basedir, 'channels.txt')) as f:
    contents = [ l.strip() for l in f.readlines() if not l.startswith('#') ]
    channels = [ int(l.split(':', 1)[0]) for l in contents ]
    for i, ch in enumerate(channels):
        ch_index[ch] = i

pattern = '*-[0-9][0-9][0-9].jpg' if orig_only else '*.jpg'
evalfiles = glob.glob(join(evaldir, '*', pattern))
trainfiles = glob.glob(join(traindir, '*', pattern))

random.shuffle(evalfiles)
random.shuffle(trainfiles)

with open(join(basedir, 'evaluate.txt'), 'w') as manifest:
    for f in evalfiles:
        ch = int(basename(dirname(f)))
        if ch in ch_index:
            manifest.write('%s:%d\n' % (f, ch_index[ch]))

with open(join(basedir, 'train.txt'), 'w') as manifest:
    for f in trainfiles:
        ch = int(basename(dirname(f)))
        if ch in ch_index:
            manifest.write('%s:%d\n' % (f, ch_index[ch]))
