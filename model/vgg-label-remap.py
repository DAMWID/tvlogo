#!/usr/bin/python

import getopt
import os
import random
import sys
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from os.path import isdir, isfile, join, basename, dirname
from PIL import Image
import vgg

VGG16_LAYOUT = ((224, 224, 3), ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)), (4096, 4096), 0)
CUSTOM_LAYOUT = ((224, 224, 3), ((32, 32), (64, 64), (128, 128)), (1024,), 0)
SIMPLE_LAYOUT = ((224, 224, 3), ((32, 32), (64, 64)), (1024,), 0)
LINEAR_LAYOUT = ((224, 224, 3), (), (), 0)

basedir = '.'
layout = list(CUSTOM_LAYOUT)

class Labels(object):
    """Class to maintain label definitions"""

    def __init__(self, labelfile):
        with open(labelfile) as f:
            contents = [ l.strip() for l in f.readlines() if not l.startswith('#') ]
            self.count = [ int(l.split(':', 2)[0]) for l in contents if l.split(':', 2)[1] == 'max' ][0]
            self.db = [('_', 'unknown')] * self.count
            self.lut = {}
            for l in contents:
                i, ch, name = l.split(':', 2)
                if ch == 'max':
                    continue
                self.db[int(i)] = (ch, name)
                self.lut[ch] = int(i)
            self.empty = [ i for i, info in enumerate(self.db) if info[0] == '_' ]

if len(sys.argv) < 3:
    print('Usage: %s <old_label_map> <new_label_map>' % sys.argv[0])
    sys.exit()

os.chdir(basedir)

old_labels = Labels(sys.argv[1])
new_labels = Labels(sys.argv[2])

remap = [0] * new_labels.count
for i in range(new_labels.count):
    ch = new_labels.db[i][0]
    if ch in old_labels.lut:
        remap[i] = old_labels.lut[ch]
    else:
        remap[i] = old_labels.empty[0]
        if len(old_labels.empty) > 1:
            old_labels.empty = old_labels.empty[1:]

print(remap)

layout[0] = (160, 160, 3)
layout[-1] = old_labels.count

attr_i = 'x'.join([ str(l) for l in layout[0] ])
attr_c = '_'.join([ str(l[0]) for l in layout[1] ])
attr_f = '_'.join([ str(l) for l in layout[2] ])
attr = '-'.join([ s for s in (attr_i, attr_c, attr_f, str(layout[-1])) if len(s) > 0 ])

model_param_path = join(basedir, 'pretrained', 'tvlogo-vgg-%s.npy' % attr)
model = vgg.VGG(layout, model_param_path)
model.save(model_param_path.replace('.npy', '-new.npy'), remap=remap)
