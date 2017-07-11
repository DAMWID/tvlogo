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
AUGMENT = 0
FLIP = False

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:fij:", ['help', 'augment=', 'flip', 'imagenet', 'jitter='])
    except getopt.GetoptError, msg:
        raise Usage(msg)

    if len(args) != 2 or not isdir(args[0]):
        raise Usage(None)
    snapdir, logodir = args

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            raise Usage(None)
        elif opt in ('-a', '--augment'):
            AUGMENT = int(arg)
            if AUGMENT < 0:
                AUGMENT = 0
        elif opt in ('-f', '--flip'):
            FLIP = True
        elif opt in ('-i', '--imagenet'):
            CROP_X, CROP_Y, CROP_W, CROP_H = (0, 0, 224, 112)
        elif opt in ('-j', '--jitter'):
            JITTER = int(arg)
            if JITTER < 0:
                JITTER = 0

    j = JITTER * 2 + 1
    c = (j**2) / 2

    if AUGMENT >= j**2:
        raise Usage('augment must less than (2*JITTER+1)^2')
except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s OPTION SNAPDIR LOGODIR' % sys.argv[0])
    sys.exit(2)


# Create a bigger array with a margin of JITTER pixel on each side
height, width = HEIGHT + JITTER * 2, WIDTH + JITTER * 2
CROP_X, CROP_Y  = CROP_X + JITTER, CROP_Y + JITTER

img = np.zeros((height, width, 3), dtype=np.uint8)

for f in glob.glob(join(snapdir, '*', '*.jpg')):
    logo_file = join(logodir, relpath(f, snapdir))
    try:
        os.makedirs(dirname(logo_file))
    except OSError:
        pass

    # copy the image to the center of the bigger array
    img[JITTER:JITTER+HEIGHT, JITTER:JITTER+WIDTH, :] = np.asarray(Image.open(f).resize((WIDTH, HEIGHT), Image.BICUBIC), dtype=np.uint8)

    # random select samples in a j*j square
    # including the original one if not already created
    offsets = range(c) + range(c+1, j**2)
    random.shuffle(offsets)
    if not isfile(logo_file):
        offsets = [c, ] + offsets
        n = AUGMENT + 1
    else:
        n = AUGMENT

    for off in offsets:
        dx, dy = off % j - JITTER, off / j - JITTER
        suffix = '' if dx == 0 and dy == 0 else ('+%d+%d' % (dx, dy)).replace('+-', '-')
        filename = logo_file.replace('.jpg', '%s.jpg' % suffix)
        if isfile(filename):
            continue

        crop_x, crop_y = CROP_X + dx, CROP_Y + dy
        crop = np.append(img[crop_y:crop_y+CROP_H, crop_x:crop_x+CROP_W, :], img[crop_y:crop_y+CROP_H, width-crop_x-CROP_W:width-crop_x, :], axis=0)
        Image.fromarray(crop).save(filename, quality=100)
        if FLIP:
            crop = np.append(img[crop_y:crop_y+CROP_H, width-crop_x-CROP_W:width-crop_x, :], img[crop_y:crop_y+CROP_H, crop_x:crop_x+CROP_W, :], axis=0)
            Image.fromarray(crop).save(filename.replace('.jpg', '-flip.jpg'), quality=100)
        n -= 1
        if n == 0:
            break
