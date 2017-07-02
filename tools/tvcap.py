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

IR_IP = "192.168.1.203"
IR_PORT = 7001

MQTT_SERVER = "192.168.1.4"

BGCTV_CMND_TOPIC = "cmnd/BGCTV/POWER"
BGCTV_STAT_TOPIC = "stat/BGCTV/POWER"
BGCTV_STREAM = 'udp://239.255.42.42:5004'

SNAPSHOTS_LIMIT = 5000
SNAPSHOTS_BATCH = 200
KEYFRAMES_PER_SNAP = 5
DELTA_THRESHOLD = 768
STALL_THRESHOLD = 20
MAX_STALL_SECONDS = 20
MAX_SKIP_SECONDS = 60
MIN_BATCH_FRAMES = 10

WIDTH = 640
HEIGHT = 360
LOGO_CROP_W = 160
LOGO_CROP_H = 80
LOGO_CROP_X = 20
LOGO_CROP_Y = 4

use_keyframe = True

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
snap_dir = join(basedir, 'snap')

crop = np.array(((0, 0), (LOGO_CROP_W, LOGO_CROP_H)))
base0 = np.array((LOGO_CROP_X, LOGO_CROP_Y))
base1 = np.array((WIDTH-LOGO_CROP_X-LOGO_CROP_W, LOGO_CROP_Y))
region0 = (crop + base0).ravel()
region1 = (crop + base1).ravel()

class BGCTV(object):
    """Class to control BGCTV"""

    def __init__(self, channels_file):
        self.ir_data = {}
        self.channels = {}
        self.ch_snaps = {}

        # UDP socket used to send IR data packet to IR transmitter
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_connected = False
        self.sw_online = False
        self.powered_on = False
        self.current_ch = None

        self.mqtt_client.connect(MQTT_SERVER, 1883, 60)
        while not self.mqtt_connected:
            self.mqtt_client.loop(timeout=0.1, max_packets=1)

        keys_dir = join(expanduser("~"), "etc", "irkeys")
        key_idx = len(join(keys_dir, 'key_'))
        for fn in glob.glob(join(keys_dir, 'key_*.dat')):
            if not isfile(fn):
                continue
            key = fn[key_idx:-4]
            with open(fn, 'r') as f:
                msg = f.read()
                self.ir_data[key] = msg

        with open(channels_file) as f:
            content = f.readlines()
        content = [ x.strip() for x in content if not x.startswith('#') ]
        for x in content:
            ch, name = x.split(':', 1)
            self.channels[ch] = name
            ch_dir = join(snap_dir, ch)
            try:
                os.makedirs(ch_dir)
            except OSError:
                pass
            #self.ch_snaps[ch] = len([ f for f in listdir(ch_dir) if f.startswith("snap-") and f.endswith(".jpg") ])
            self.ch_snaps[ch] = len(glob.glob(join(ch_dir, 'snap-*.jpg')))

    def ir_send(self, key):
        self.sock.sendto(self.ir_data[key], (IR_IP, IR_PORT))

    def on_mqtt_connect(self, mqtt_client, userdata, flags, rc):
        #print("Connected with result code " + str(rc))
        self.mqtt_connected = True
      
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        mqtt_client.subscribe(BGCTV_STAT_TOPIC)
        mqtt_client.publish(BGCTV_CMND_TOPIC, "9")
    
    def on_mqtt_message(self, mqtt_client, userdata, msg):
        #print(msg.topic + " " + str(msg.payload))
    
        if msg.topic == BGCTV_STAT_TOPIC:
            self.sw_online = True
            if msg.payload == "ON":
                self.powered_on = True
            elif msg.payload == "OFF":
                self.powered_on = False

    def power_on(self):
        self.mqtt_client.publish(BGCTV_CMND_TOPIC, "ON")
    
    def power_off(self):
        self.mqtt_client.publish(BGCTV_CMND_TOPIC, "OFF")
    
    def set_channel(self, channel):
        print('CH %s: [%s]' % (channel, self.channels[channel]))
        self.ir_send('quit')
        time.sleep(1)
        for num in str(channel):
            self.ir_send(num)
            time.sleep(.5)
        self.ir_send('enter')
        self.current_ch = channel

    def wait_online(self):
        timeout = 5
        while not self.sw_online and timeout > 0:
            self.mqtt_client.loop(timeout=1, max_packets=1)
            timeout -= 1

        if timeout == 0:
            return False

        if not self.powered_on:
            print('Powering on BGCTV...')
            self.power_on()
            time.sleep(30)
        return True

    def next_channel(self):
        l = [x[0] for x in sorted(self.ch_snaps.items(), key=lambda d: d[1]) if x[1] < limit]
        if len(l) == 0:
            return (0, '')
        ch = l[random.randint(0, len(l)/4)]
        self.set_channel(ch)
        b = min(limit - self.ch_snaps[ch], batch)
        return (str(ch), b)

    def captured(self, num):
        self.ch_snaps[self.current_ch] += num

tv = BGCTV(channel_file)

success = tv.wait_online()
if not success:
    print('BGCTV Power Control not online, exit.')
    sys.exit()

sys.path.append(join(basedir, 'model'))
import tvlogo

logo = tvlogo.tvlogo(basedir)

try:
    while True:
        ch, b = tv.next_channel()
        if b == 0:
            break
    
        container = av.open(BGCTV_STREAM)

        fps = 1 if use_keyframe else int(container.streams.video[0].average_rate)

        wait_frames = fps * 30

        localtime = time.localtime(time.time())
        timestamp = '%02d%02d%02d%02d' % (localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)

        frames = 0
        matches = 0
        captured = 0
        stalled = 0
        skipped = 0
        total_skipped = 0
        prev_img_a = np.array([0], dtype=np.uint8)
        for frame in container.decode(video=0):
            if use_keyframe and not frame.key_frame:
                continue

            im = frame.reformat(width=WIDTH, height=HEIGHT).to_image()

            if matches < 5:
                result = logo.classify(im)
                print(result)
                if result[0][0] == ch and result[0][1] > 0.5:
                    matches += 1
                wait_frames -= 1
                if wait_frames == 0:
                    break
                continue

            frames += 1
    
            img_a = np.append(np.asarray(im.crop(region0), dtype=np.uint8), np.asarray(im.crop(region1), dtype=np.uint8), axis=0)

            diff = np.subtract(img_a.astype('float32'), prev_img_a.astype('float32'))
            diff[LOGO_CROP_H:, :, :] /= 2
            d = (np.sum(np.square(diff)) / img_a.size).astype('uint32')
            if d > DELTA_THRESHOLD:
                captured += 1
                im.save(join(snap_dir, ch, 'snap-%s-%03d.jpg' % (timestamp, captured)), quality=100)
                prev_img_a = img_a
                stalled = 0
                skipped = 0
            elif d < STALL_THRESHOLD:
                stalled += 1
                skipped += 1
                total_skipped += 1
            else:
                stalled = 0
                skipped += 1
                total_skipped += 1
    
            ratio = captured * 100 / b
            print('%4d: [%s%s] [Delta: %5d] [Skipped: %4d] [Stall: %4d]' % (captured, '=' * ratio, '.' * (100 - ratio), d, total_skipped, stalled), end='\r')
            sys.stdout.flush()
            if frames % fps == 0:
                tv.ir_send('quit')
            if captured == b or stalled == fps * MAX_STALL_SECONDS or skipped == fps * MAX_SKIP_SECONDS:
                if captured < b and captured < MIN_BATCH_FRAMES:
                    for f in glob.glob(join(snap_dir, ch, 'snap-%s-*.jpg' % timestamp)):
                        os.remove(f)
                else:
                    tv.captured(captured)
                print('')
                sys.stdout.flush()
                break
    
        time.sleep(1)
except BaseException:
    pass

