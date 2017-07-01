#!/usr/bin/python

import getopt
import glob
import os
import random
import sys
from PIL import Image
from os.path import isdir, isfile, join, expanduser, basename, dirname, relpath
import numpy as np

WIDTH = 640
HEIGHT = 360
CROP_W = 160
CROP_H = 80
CROP_X = 20
CROP_Y = 4
JITTER = 0
AUGMENT = 1

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:ij:", ['help', 'augment=', 'imagenet', 'jitter='])
    except getopt.GetoptError, msg:
        raise Usage(msg)

    if len(args) != 1 or not isdir(args[0]):
        raise Usage(None)
    basedir = args[0]

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            raise Usage(None)
        elif opt in ('-a', '--augment'):
            AUGMENT = int(arg)
            if AUGMENT < 1:
                AUGMENT = 1
        elif opt in ('-i', '--imagenet'):
            CROP_X, CROP_Y, CROP_W, CROP_H = (0, 0, 224, 112)
        elif opt in ('-j', '--jitter'):
            JITTER = int(arg)
            if JITTER < 0:
                JITTER = 0

    if JITTER > CROP_X:
        CROP_X = JITTER
    if JITTER > CROP_Y:
        CROP_Y = JITTER

    if AUGMENT > (2*JITTER+1)**2:
        raise Usage('augment must less than or equals to (2*jitter+1)^2')
except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s OPTION [dir]' % sys.argv[0])
    sys.exit(2)

snapdir = join(basedir, 'snap')
logodir = join(basedir, 'logo')

snapfiles = glob.glob(join(snapdir, '*', 'snap-*.jpg'))

img_a = np.zeros((CROP_H*2, CROP_W, 3), dtype=np.uint8)

for f in snapfiles:
    im = Image.open(f)
    logo_file = f.replace('snap', 'logo')
    try:
        os.makedirs(dirname(logo_file))
    except OSError:
        pass

    img = np.asarray(im.resize((WIDTH, HEIGHT), Image.BICUBIC), dtype=np.uint8)

    j_size = JITTER * 2 + 1
    offsets = random.sample(range(j_size**2), AUGMENT)
    for i, off in enumerate(offsets):
        crop_x = CROP_X - JITTER + off % j_size
        crop_y = CROP_Y - JITTER + off / j_size
        img_a[:CROP_H, :, :] = img[crop_y:crop_y+CROP_H, crop_x:crop_x+CROP_W, :]
        img_a[CROP_H:, :, :] = img[crop_y:crop_y+CROP_H, WIDTH-crop_x-CROP_W:WIDTH-crop_x, :]
        Image.fromarray(img_a).save(logo_file.replace('.jpg', '-%d.jpg' % i), quality=100)

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

with open(join(basedir, 'test.txt'), 'w') as test:
    for f in testfiles:
        ch = int(basename(dirname(f)))
        test.write('%s:%d\n' % (f, ch_index[ch]))

with open(join(basedir, 'train.txt'), 'w') as train:
    for f in trainfiles:
        ch = int(basename(dirname(f)))
        train.write('%s:%d\n' % (f, ch_index[ch]))
