#!/usr/bin/python

import glob
import re
import os
import sys
from PIL import Image, ImageFont, ImageDraw
from os import listdir
from os.path import isdir, isfile, join, expanduser, basename
import numpy as np

MIN_BATCH = 10

if len(sys.argv) < 2:
    print('Usage: %s <DIR> ...' % sys.argv[0])
    sys.exit()

chdirs = [ re.sub('/$', '', d) for d in sys.argv[1:] if isdir(d) ]

for chdir in chdirs:
    batch = [ f.split('-')[1] for f in listdir(chdir) if (f.startswith("snap-") or f.startswith("logo-")) and f.endswith(".jpg")]
    batch = list(set(batch))

    if len(batch) == 0:
        continue

    for i in range(len(batch)):
        files = glob.glob(join(chdir, '*-%s-*.jpg' % batch[i]))
        if len(files) < MIN_BATCH:
            #print("%s: %d files" % (batch[i], len(files)))
            for f in files:
                print('Deleting %s' % f)
                os.remove(f)
