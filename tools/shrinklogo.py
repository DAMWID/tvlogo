#!/usr/bin/python

import glob
import os
import sys
from PIL import Image
from os.path import isfile, join, expanduser, basename
import numpy as np

WIDTH = 80
HEIGHT = 80
LOGO_W = 40
LOGO_H = 40

OFF_X = 10
OFF_Y = 12

region0 = (OFF_X, OFF_Y, OFF_X+LOGO_W, OFF_Y+LOGO_H/2)
region1 = (OFF_X, OFF_Y+LOGO_H, OFF_X+LOGO_W, OFF_Y+LOGO_H+LOGO_H/2)

dirs = sys.argv[1:]

if len(dirs) == 0:
    print('Usage: %s <DIR> ...' % sys.argv[0])
    sys.exit()

for d in dirs:
    files = glob.glob(join(d, 'logo-*.jpg'))

    if len(files) == 0:
        print('%s: no image to operate.' % d)
        continue

    for f in files:
        a = np.zeros((LOGO_H, LOGO_W, 3), dtype=np.uint8)
        im = Image.open(f).resize((WIDTH, HEIGHT), Image.BICUBIC)
        a[:LOGO_H/2, :, :] = np.asarray(im.crop(region0), dtype=np.uint8)
        a[LOGO_H/2:, :, :] = np.asarray(im.crop(region1), dtype=np.uint8)
        Image.fromarray(a).save(f, quality=100)
        #Image.open(f).resize((WIDTH, HEIGHT), Image.BICUBIC).save(f, quality=100)
