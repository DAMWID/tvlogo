#!/usr/bin/python

import getopt
import glob
import os
import random
import shutil
import sys
import tempfile
from PIL import Image
from os.path import isdir, isfile, join, expanduser, basename, dirname, relpath
import numpy as np

WIDTH = 640
HEIGHT = 360
CROP_W = 160
CROP_H = CROP_W / 2
CROP_X = 20
CROP_Y = 4
JITTER_X = 16
JITTER_Y = 9

augment = 0
allow_jitter = False
allow_flip = False
allow_scale = False
do_quarter = False
do_blend = False
allow_invalid = False

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:bfijnqs", ['help', 'augment=', 'blend', 'flip', 'imagenet', 'jitter', 'invalid', 'quarter', 'scale'])
    except getopt.GetoptError, msg:
        raise Usage(msg)

    if len(args) != 2 or not isdir(args[0]):
        raise Usage(None)
    snapdir, logodir = args

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            raise Usage(None)
        elif opt in ('-a', '--augment'):
            augment = int(arg)
            if augment < 0:
                augment = 0
        elif opt in ('-b', '--blend'):
            do_blend = True
        elif opt in ('-f', '--flip'):
            allow_flip = True
        elif opt in ('-i', '--imagenet'):
            CROP_X, CROP_Y, CROP_W, CROP_H = (0, 0, 224, 112)
        elif opt in ('-j', '--jitter'):
            allow_jitter = True
        elif opt in ('-n', '--invalid'):
            allow_invalid = True
        elif opt in ('-q', '--quarter'):
            do_quarter = True
        elif opt in ('-s', '--scale'):
            allow_scale = True

except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s OPTION SNAPDIR LOGODIR' % sys.argv[0])
    print('Option:')
    print('\t-a --augment=n\taugment dataset by num times for each example')
    print('\t-b --blend\tenable blending multiple images when doing dataset augment')
    print('\t-f --flip\tenable filping image when doing dataset augment')
    print('\t-i --imagenet\tuse the same image size as imagenet (224x224x3)')
    print('\t-j --jitter\tenable random shift in a small range when doing dataset augment')
    print('\t-n --invalid\tgenerate invlalid logo crop')
    print('\t-q --quarter\tuse quarter size of the image (80*80*3)')
    print('\t-s --scale\tenable scaling image by a small factor when doing dataset augment')
    print('\t-h --help\tshow this message')
    sys.exit(2)

INV_REGION = np.array([[-CROP_W/2, -CROP_H], [CROP_W/2, CROP_H]], dtype=np.float32) / (WIDTH, HEIGHT) + 0.5
CROP_ORIGIN = np.array([CROP_X, CROP_Y], dtype=np.float32) / (WIDTH, HEIGHT)
CROP_REGION = np.array([[0, 0], [CROP_W, CROP_H]], dtype=np.float32) / (WIDTH, HEIGHT)
MAX_JITTER = np.array([JITTER_X, JITTER_Y], dtype=np.float32) / (WIDTH, HEIGHT)
SCALE_RATIO = 1.0/20

out_size = (CROP_W/2, CROP_H) if do_quarter else (CROP_W, CROP_H*2)

invdir = join(logodir, 'inv')
try:
    os.makedirs(invdir)
except OSError:
    pass

channels = [ d for d in os.listdir(snapdir) if isdir(join(snapdir, d)) ]

allfiles = glob.glob(join(snapdir, '*', '*.jpg'))

if do_blend:
    tempdir = tempfile.mkdtemp(prefix='blend-')

    for ch in channels:
        snapfiles = glob.glob(join(snapdir, ch, '*.jpg'))
        total = len(snapfiles)
        if total == 0:
            continue

        blenddir = join(tempdir, ch)
        try:
            os.makedirs(blenddir)
        except OSError:
            pass

        for n in range(int(np.log2((total + 99) / 100))+1):
            random.shuffle(snapfiles)
            num = total / (2**n)
            for p in range(2**n):
                blend_sum = np.zeros((1, 1, 3), dtype=np.int32)
                for i in range(num*p, num*(p+1)):
                    blend_sum = np.asarray(Image.open(snapfiles[i]), dtype=np.uint8) + blend_sum
                blend = (blend_sum / num).astype('uint8')
                Image.fromarray(blend).save(join(blenddir, 'blend-%s-%d-%d.jpg' % (ch, num, p)), quality=100)

    allfiles += glob.glob(join(tempdir, '*', '*.jpg'))

for f in allfiles:
    ch = basename(dirname(f))
    logo_file = join(logodir, ch, basename(f))
    inv_file = join(invdir, basename(f))
    try:
        os.makedirs(dirname(logo_file))
    except OSError:
        pass

    try:
        im = Image.open(f)

        do_invalid = random.randint(0, len(channels)/augment/2) == 0
        if do_invalid and ch.isdigit():
            invalid = (INV_REGION * im.size).astype('int')
            im.crop(invalid.ravel()).resize(out_size, Image.BICUBIC).save(inv_file, quality=100)

        # extent to a larger image with a margin of jitter pixel on each side
        new_size = tuple((im.size * (1 + MAX_JITTER * 2)).astype('int'))
        rect = tuple((np.array((-MAX_JITTER, 1+MAX_JITTER))*im.size).astype('int').ravel())
        img = im.transform(new_size, Image.EXTENT, rect)
    except:
        continue


    i = 1 if isfile(logo_file) else 0
    while True:
        if i == 0:
            do_jitter = False
            do_scale = False
            do_flip = False
        else:
            do_jitter = allow_jitter
            do_scale = random.choice((True, False, False)) if allow_scale else False
            do_flip = random.choice((True, False)) if allow_flip else False

        origin = ((CROP_ORIGIN + MAX_JITTER) * im.size).astype('int')

        scale_to_3_4 = random.choice((True, False, False)) if do_scale else False
        if scale_to_3_4:
            scale_x = random.uniform(0.75-SCALE_RATIO, 0.75+SCALE_RATIO)
            jitter_ratio = (random.uniform(0, 1), random.uniform(0, 1)) if do_jitter else (0, 0)
        else:
            scale_x = random.uniform(1.0-SCALE_RATIO, 1.0+SCALE_RATIO) if do_scale else 1.0
            jitter_ratio = (random.uniform(-1, 1), random.uniform(-1, 1)) if do_jitter else (0, 0)

        scale_y = random.uniform(1-SCALE_RATIO, 1+SCALE_RATIO) if do_scale else 1.0

        crop = (CROP_REGION * (scale_x, scale_y) * im.size).astype('int')

        jitter = (MAX_JITTER * jitter_ratio * im.size).astype('int')

        # +------------------------------------------------
        # |   MAX_JITTER
        # |   +--------------------------------------------
        # |   |    <jx>
        # |   |   o---+
        # |   |   |<jy>
        # |   |   +   +----------------+
        # |   |       |                |
        # |   |       |                |
        # |   |       |                |
        # |   |       |                |
        # |   |       +----------------+
        # |   |
        region0 = origin + crop + jitter
        region1 = np.array([[-region0[1][0], region0[0][1]], [-region0[0][0], region0[1][1]]]) + (img.width, 0)

        w, h = list(region0[1] - region0[0])
        jx, jy = list(jitter)

        suffix = ''
        if (scale_x, scale_y) != (1.0, 1.0):
            suffix += '-%dx%d' % (w, h)

        if (jx, jy) != (0, 0):
            suffix += ('+%d+%d' % (jx, jy)).replace('+-', '-')

        if do_flip:
            suffix += '-flip'

        filename = logo_file.replace('.jpg', '%s.jpg' % suffix)
        if isfile(filename):
            continue

        crop0, crop1 = img.crop(region0.ravel()), img.crop(region1.ravel())
        if do_flip:
            crop0, crop1 = crop1, crop0

        logo = Image.new(im.mode, (w, h*2), (0,0,0))
        logo.paste(crop0, (0, 0, w, h))
        logo.paste(crop1, (0, h, w, h*2))

        out_size = (CROP_W/2, CROP_H) if do_quarter else (CROP_W, CROP_H*2)
        logo.resize(out_size, Image.BICUBIC).save(filename, quality=100)

        i += 1
        if i == augment + 1:
            break

if do_blend:
    shutil.rmtree(tempdir)
