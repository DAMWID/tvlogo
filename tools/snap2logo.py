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
SCALE = False
SCALE_RATIO = 1/20.0
QUARTER = False

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:fij:s", ['help', 'augment=', 'flip', 'imagenet', 'jitter=', 'scale='])
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
        elif opt in ('-q', '--quarter'):
            QUARTER = True
        elif opt in ('-s', '--scale'):
            SCALE = True

    j = JITTER * 2 + 1
    c = (j**2) / 2

    if AUGMENT >= j**2:
        raise Usage('augment must less than (2*JITTER+1)^2')
except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s OPTION SNAPDIR LOGODIR' % sys.argv[0])
    print('Option:')
    print('\t-a --augment=n\taugment dataset by num times for each example')
    print('\t-f --flip\tenable filping image when doing dataset augment')
    print('\t-i --imagenet\tuse the same image size as imagenet (224x224x3)')
    print('\t-j --jitter=n\trandom shift in a (2*num+1)*(2*num+1) square')
    print('\t-q --quarter\tuser quarter size of the image (80*80*3)')
    print('\t-s --scale\tenable scaling image by a small factor when doing dataset augment')
    print('\t-h --help\tshow this message')
    sys.exit(2)

SCALE_DW = int(WIDTH * SCALE_RATIO)
SCALE_DH = int(HEIGHT * SCALE_RATIO)

# Create a bigger array with a margin of JITTER pixel on each side
img = np.zeros((HEIGHT+JITTER*2+SCALE_DH, WIDTH+JITTER*2+SCALE_DW, 3), dtype=np.uint8)
CROP_X, CROP_Y  = CROP_X + JITTER, CROP_Y + JITTER

for f in glob.glob(join(snapdir, '*', '*.jpg')):
    logo_file = join(logodir, relpath(f, snapdir))
    try:
        os.makedirs(dirname(logo_file))
    except OSError:
        pass

    # rancom select 1/4 images to scale
    scale = random.choice((True, False, False, False)) if SCALE else False
    if scale:
        dw = random.randint(-SCALE_DW, SCALE_DW)
        dh = random.randint(-SCALE_DH, SCALE_DH)
    else:
        dw, dh = 0, 0

    scale_w, scale_h = WIDTH + dw, HEIGHT + dh

    height, width = scale_h + JITTER * 2, scale_w + JITTER * 2
    # copy the image to the center of the bigger array
    img[JITTER:JITTER+scale_h, JITTER:JITTER+scale_w, :] = np.asarray(Image.open(f).resize((scale_w, scale_h), Image.BICUBIC), dtype=np.uint8)

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

        flip = random.choice((True, False)) if FLIP else False
        if flip:
            crop = np.append(img[crop_y:crop_y+CROP_H, width-crop_x-CROP_W:width-crop_x, :], img[crop_y:crop_y+CROP_H, crop_x:crop_x+CROP_W, :], axis=0)
        else:
            crop = np.append(img[crop_y:crop_y+CROP_H, crop_x:crop_x+CROP_W, :], img[crop_y:crop_y+CROP_H, width-crop_x-CROP_W:width-crop_x, :], axis=0)

        im = Image.fromarray(crop)
        if QUARTER:
            im = im.resize((im.widht/2, im.height/2), Image.BICUBIC)
        im.save(filename, quality=100)

        n -= 1
        if n == 0:
            break
