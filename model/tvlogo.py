#!/usr/bin/python

import getopt
import os
import random
import sys
import time
import numpy as np
from os.path import isdir, isfile, join
from PIL import Image
import vgg

class Labels(object):
    """Class to maintain label definitions"""

    def __init__(self, labelfile, labels):
        with open(labelfile) as f:
            contents = [ l.strip() for l in f.readlines() if not l.startswith('#') ]
            self.count = labels if labels > len(contents) else len(contents)
            self.db = [('_', 'Unknow')] * self.count
            for i, l in enumerate(contents):
                self.db[i] = l.split(':', 1)

class tvlogo(object):
    """Class to classify a given image to known TV channels name"""

    layout = ((160, 160, 3), ((32, 32), (64, 64), (128, 128)), (1024,), 1000)
    resize = (640, 360)
    tl_box = (20, 4, 180, 84)
    tr_box = (460, 4, 620, 84)

    def __init__(self, confdir='.'):
        self.label = Labels(join(confdir, 'channels.txt'), self.layout[-1])

        attr_i = 'x'.join([ str(l) for l in self.layout[0] ])
        attr_c = '_'.join([ str(l[0]) for l in self.layout[1] ])
        attr_f = '_'.join([ str(l) for l in self.layout[2] ])
        attr = '-'.join([ s for s in (attr_i, attr_c, attr_f, str(self.layout[-1])) if len(s) > 0 ])

        model_param_path = join(confdir, 'pretrained', 'tvlogo-vgg-%s.npy' % attr)
        self.model = vgg.VGG(self.layout, model_param_path)

    def classify(self, image, top=5):
        if image.size != (640, 360):
            image = image.resize(self.resize, Image.BICUBIC)
        crop = np.append(np.asarray(img.crop(self.tl_box), dtype=np.uint8), np.asarray(img.crop(self.tr_box), dtype=np.uint8), axis=0)
        prob = self.model.infer(crop.reshape(-1, crop.size))
        return [ (self.label.db[i][0], prob[0][i]) for i in np.argsort(prob[0])[:-(top+1):-1] ]
