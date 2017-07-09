#!/usr/bin/python

from __future__ import print_function
import av
import glob
import subprocess as proc
import signal
import socket
import threading
import time
import os
import random
import select
import sys
import time
import tty
import termios
from os import listdir
from os.path import isfile, isdir, join, expanduser
import paho.mqtt.client as mqtt
from PIL import Image
import numpy as np
import bgctv

SNAPSHOTS_LIMIT = 5000
SNAPSHOTS_BATCH = 200
MAX_STALL_SECONDS = 20
MAX_SKIP_SECONDS = 60
MIN_BATCH_FRAMES = 10

imgconv = ((640, 360), (160, 80, 20, 4))

argc = len(sys.argv)
if argc < 2 or argc > 4 or not isdir(sys.argv[1]) or (argc > 2 and not sys.argv[2].isdigit()) or (argc > 3 and not sys.argv[3].isdigit()):
    print('Usage: %s <DIR> <LIMIT> <BATCH>' % sys.argv[0])
    sys.exit()

basedir = sys.argv[1]

if len(sys.argv) > 2:
    limit = int(sys.argv[2])
else:
    limit = SNAPSHOTS_LIMIT

if len(sys.argv) > 3:
    batch = int(sys.argv[3])
else:
    batch = SNAPSHOTS_BATCH

channel_file = join(basedir, 'channels.txt')
snapdir = join(basedir, 'snap')

tv = bgctv.BGCTV(channel_file, imgconv)

# score each channel by snaps they already have
for i, ch in enumerate(tv.channels):
    if ch[0].isdigit():
        try:
            os.makedirs(join(snapdir, ch[0]))
        except OSError:
            pass

        n = len(glob.glob(join(snapdir, ch[0], 'snap-*.jpg')))
        tv.score(i, n)

success = tv.wait_online()
if not success:
    print('BGCTV Power Control not online, exit.')
    sys.exit()

sys.path.append(join(basedir, 'model'))
import tvlogo

logo = tvlogo.tvlogo(basedir)

try:
    while True:
        ch_idx, n = tv.next_channel_by_score(tv.PRIO_MIN)
        ch = tv.channels[ch_idx][0]
        if n >= limit:
            break
    
        b = limit - n
        if b > batch:
            b = batch

        tv.batch_start(1, tv.OUTPUT_LOGO_CROP, tv.MODE_KEYFRAME_MIXING)
        matches = 0

        try:
            while matches < 3:
                bx, _ = tv.batch_get()
                if bx is None:
                    break

                result = logo.classify(bx)
                bx = None
                print(result)
                if result[0][0] == ch:
                    matches += 1
                else:
                    matches = 0
        except:
            break
        finally:
            tv.batch_stop()

        time.sleep(1)

        localtime = time.localtime(time.time())
        timestamp = '%02d%02d%02d%02d' % (localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)

        tv.batch_start(1, tv.OUTPUT_RAW_IMAGE, tv.MODE_KEYFRAME_DELTA, max_stall=MAX_STALL_SECONDS, max_skip=MAX_SKIP_SECONDS)
        #tv.batch_start(1, tv.OUTPUT_RAW_IMAGE, tv.MODE_KEYFRAME_ANY)
        captured = 0
        try:
            print('%4d: [%s]' % (captured, '.' * 100), end='\r')
            sys.stdout.flush()

            # skip first frame
            im, _ = tv.batch_get()

            while captured < b:
                im, _ = tv.batch_get()
                if im is None:
                    break
                captured += 1
                im.save(join(snapdir, ch, 'snap-%s-%03d.jpg' % (timestamp, captured)), quality=100)
                ratio = captured * 100 / b
                print('%4d: [%s%s]' % (captured, '=' * ratio, '.' * (100 - ratio)), end='\r')
                sys.stdout.flush()
        except:
            break
        finally:
            tv.batch_stop()
            tv.score(ch_idx, captured)
            print('')
            sys.stdout.flush()

        time.sleep(1)
except BaseException:
    pass

