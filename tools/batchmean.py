#!/usr/bin/python

import getopt
import glob
import re
import sys
from PIL import Image, ImageFont, ImageDraw
from os import listdir
from os.path import isdir, isfile, join, expanduser, basename
import numpy as np

COL = 8
title_h = 32
first = False
mean = False
img_per_batch = 1

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

try:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h1mc:", ['help', 'first', 'mean', 'col'])
    except getopt.GetoptError, msg:
        raise Usage(msg)

    if len(args) < 1:
        raise Usage(None)
    chdirs = [ re.sub('/$', '', d) for d in args if isdir(d) ]

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            raise Usage(None)
        elif opt in ('-1', '--first'):
            first = True
        elif opt in ('-m', '--mean'):
            mean = True
        elif opt in ('-c', '--col'):
            COL = int(arg)
except Usage, err:
    if err.msg:
        print >> sys.stderr, err.msg
    print('Usage: %s OPTION [dir] ...' % sys.argv[0])
    sys.exit(2)

if not mean and not first:
    mean = True

if mean and first:
    img_per_batch = 2;

COL *= img_per_batch

font = ImageFont.truetype(join(expanduser("~"), "MC360.ttf"), 24)

for chdir in chdirs:
    batch = [ f.split('-')[1] for f in listdir(chdir) if (f.startswith("snap-") or f.startswith("logo-")) and f.endswith(".jpg")]
    batch = list(set(batch))
    batch.sort()

    if len(batch) == 0:
        continue

    num = img_per_batch * len(batch)
    row = (num + COL - 1) / COL
    if row == 1:
        col = num
    else:
        col = COL

    f = glob.glob(join(chdir, '*-%s-*.jpg' % batch[0]))[0]
    width, height = Image.open(f).size

    img_a = np.zeros(((height + title_h) * row, width * col, 3), dtype=np.uint8)

    for i in range(len(batch)):
        files = glob.glob(join(chdir, '*-%s-*.jpg' % batch[i]))
        files.sort()

        r = i * img_per_batch / col
        c = (i * img_per_batch) % col

        row = (height + title_h) * r

        if first:
            img_a[row:row+height, width*c:width*(c+1), :] = np.array(Image.open(files[0])).astype('uint8')
            c += 1

        if mean:
            s = np.zeros((height, width, 3), dtype=np.int32)
            for f in files:
                a = np.array(Image.open(f))
                s = s + a

            img_a[row:row+height, width*c:width*(c+1), :] = (s / len(files)).astype('uint8')


    im = Image.fromarray(img_a)
    draw = ImageDraw.Draw(im)
    for i in range(len(batch)):
        n = len(glob.glob(join(chdir, '*-%s-*.jpg' % batch[i])))
        r = i * img_per_batch / col
        c = (i * img_per_batch) % col
        title_y = (height + title_h) * r + height
        if first:
            title_x = width * c + 2
            draw.text((title_x, title_y), "%s [0]" % batch[i], (255,255,255), font=font)
            c += 1

        if mean:
            title_x = width * c + 2
            draw.text((title_x, title_y), "%s [%d]" % (batch[i], n), (255,255,255), font=font)

    im.save('%s.jpg' % chdir, quality=100)
    #im.show()
