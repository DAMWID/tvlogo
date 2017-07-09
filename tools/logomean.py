#!/usr/bin/python

import glob
import re
import sys
from PIL import Image
from os.path import isdir, isfile, join, expanduser
import numpy as np

if len(sys.argv) < 2:
    print('Usage: %s <DIR> ...' % sys.argv[0])
    sys.exit()

dirs = [ re.sub('/$', '', d) for d in sys.argv[1:] if isdir(d) ]

col = 10
row = (len(dirs) + col - 1)/col

f = glob.glob(join(dirs[0], '*.jpg'))[0]
width, height = Image.open(f).size

img_a = np.zeros((height * row, width * col, 3), dtype=np.uint8)
idx = 0
for d in dirs:
    files = glob.glob(join(d, '*.jpg'))
    r = idx / col
    c = idx % col
    idx += 1

    if len(files) == 0:
        print('%s: no image to operate.' % d)
        continue

    s = np.zeros((height, width, 3), dtype=np.int32)
    for f in files:
        a = np.array(Image.open(f))
        s = s + a

    img_a[height*r:height*(r+1), width*c:width*(c+1), :] = (s / len(files)).astype('uint8')


im = Image.fromarray(img_a)
im.save('logo_mean-%dx%d.jpg' % (width, height), quality=100)
#im.show()
