#!/usr/bin/python

import getopt
import glob
import os
import random
import sys
from os import listdir
from os.path import isdir, isfile, join, expanduser, abspath, relpath, basename, dirname

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

valdir = None
traindir = None
orig_only = False

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hl:ot:v:", ['help', 'label=', 'original', 'train=', 'validate='])
    except getopt.GetoptError, msg:
        raise Usage(msg)

    if len(args) != 1 or not isdir(args[0]):
        raise Usage(None)
    basedir = args[0]

    labelmap = join(basedir, 'channels.txt')

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            raise Usage(None)
        elif opt in ('-l', '--label'):
            labelmap = arg
        elif opt in ('-o', '--original'):
            orig_only = True
        elif opt in ('-t', '--train'):
            traindir = arg
        elif opt in ('-v', '--validate'):
            valdir = arg

    if valdir and not isdir(valdir):
        raise Usage('Directory %s not exists' % valdir)
    if traindir and not isdir(traindir):
        raise Usage('Directory %s not exists' % traindir)
    if not valdir and not traindir:
        raise Usage('No source directory defined')
    if not isfile(labelmap):
        raise Usage('Label map file %s not exists" % labelmap')
except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s [-t train_path ] [-v validate_path ] <target_path>' % sys.argv[0])
    sys.exit(2)

ch_index = {}
with open(labelmap) as f:
    contents = [ l.strip() for l in f.readlines() if not l.startswith('#') ]
    for l in contents:
        i, ch, name = l.split(':', 2)
        ch_index[ch] = int(i)

pattern = '*-[0-9][0-9][0-9].jpg' if orig_only else '*.jpg'

if valdir:
    valfiles = glob.glob(join(abspath(valdir), '*', pattern))
    random.shuffle(valfiles)
    with open(join(basedir, 'validate.txt'), 'w') as manifest:
        for f in valfiles:
            ch = basename(dirname(f))
            if ch in ch_index:
                manifest.write('%s:%d\n' % (f, ch_index[ch]))

if traindir:
    trainfiles = glob.glob(join(abspath(traindir), '*', pattern))
    random.shuffle(trainfiles)
    with open(join(basedir, 'train.txt'), 'w') as manifest:
        for f in trainfiles:
            ch = basename(dirname(f))
            if ch in ch_index:
                manifest.write('%s:%d\n' % (f, ch_index[ch]))
