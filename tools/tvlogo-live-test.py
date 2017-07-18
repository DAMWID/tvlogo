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

argc = len(sys.argv)
if argc < 4 or not isdir(sys.argv[1]):
    print('Usage: %s <DIR> <EPOCH> <BATCH>' % sys.argv[0])
    sys.exit()

basedir = sys.argv[1]
epoch = int(sys.argv[2])
batch = int(sys.argv[3])

channel_file = join(basedir, 'channels.map')

crop = np.array(((0, 0), (LOGO_CROP_W, LOGO_CROP_H)))
base0 = np.array((LOGO_CROP_X, LOGO_CROP_Y))
base1 = np.array((WIDTH-LOGO_CROP_X-LOGO_CROP_W, LOGO_CROP_Y))
region0 = (crop + base0).ravel()
region1 = (crop + base1).ravel()

class BGCTV(object):
    """Class to control BGCTV"""

    def __init__(self, channels_file):
        self.ir_data = {}

        # UDP socket used to send IR data packet to IR transmitter
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_connected = False
        self.sw_online = False
        self.powered_on = False

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
            contents = [ l.strip() for l in f.readlines() if not l.startswith('#') ]
            self.channels = [('_', 'Unknow')] * len(contents)
            for i, l in enumerate(contents):
                self.channels[i] = l.split(':', 1)

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

    def set_channel(self, idx):
        print("CH %s: [%s]" % (self.channels[idx][0], self.channels[idx][1]))
        self.ir_send('quit')
        time.sleep(1)
        for num in self.channels[idx][0]:
            self.ir_send(num)
            time.sleep(.5)
        self.ir_send('enter')
        self.current_ch = idx

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
        idx = random.randint(0, len(self.channels)-1)
        self.set_channel(idx)
        return self.channels[idx]

tv = BGCTV(channel_file)

fail_chart = np.zeros((len(tv.channels), 16), dtype=np.int32)

success = tv.wait_online()
if not success:
    print('BGCTV Power Control not online, exit.')
    sys.exit()

sys.path.append(join(basedir, 'model'))
import tvlogo

logo = tvlogo.tvlogo(basedir)

np.set_printoptions(threshold=np.inf)

try:
    for e in range(epoch):
        ch, name = tv.next_channel()

        container = av.open(BGCTV_STREAM)

        wait_frames = 30
        matched = 0
        remaining = batch
        failed = 0
        confirmed = False
        crop_sum = np.zeros((160, 160, 3), dtype=np.uint32)
        num = 0
        for frame in container.decode(video=0):
            im = frame.to_image().resize((WIDTH, HEIGHT), Image.BILINEAR)
            crop_sum += np.append(np.asarray(im.crop(region0), dtype=np.uint32), np.asarray(im.crop(region1), dtype=np.uint32), axis=0)
            num += 1
            if not frame.key_frame:
                continue

            tv.ir_send('quit')

            mean = (crop_sum / num).astype('uint8')

            crop_sum = np.zeros((160, 160, 3), dtype=np.uint32)
            num = 0

            result = logo.classify(mean)
            lumin = np.mean(mean)
            print(result)
            if not confirmed:
                if result[0][0] == ch:
                    matched += 1
                    if matched == 5:
                        confirmed = True
                else:
                    wait_frames -= 0
                    if wait_frames == 0:
                        break
            else:
                if result[0][0] != ch:
                    failed += 1
                    fail_chart[tv.current_ch][int(lumin/16)] += 1
                remaining -= 1
                if remaining == 0:
                    break
        if confirmed:
            print("Channel %s [%s]: test accuracy: %g" % (ch, name, 1-failed*1.0/batch))
            print(fail_chart[tv.current_ch])
        time.sleep(1)
except BaseException:
    pass

