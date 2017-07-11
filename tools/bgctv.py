#!/usr/bin/python

from __future__ import print_function
import av
import glob
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
from os.path import isfile, isdir, join, expanduser
import paho.mqtt.client as mqtt
from PIL import Image
import numpy as np

class LogoBuffer(object):
    def __init__(self, depth):
        self.depth = depth
        self.pipe = []
        self.con = threading.Condition()

    def add(self, obj, block=True):
        with self.con:
            if len(self.pipe) >= self.depth and not block:
                return False
            while len(self.pipe) >= self.depth:
                self.con.wait()
            self.pipe += [obj,]
            self.con.notify()
        return True

    def get(self, block=True):
        with self.con:
            if len(self.pipe) == 0 and not block:
                return None
            while len(self.pipe) == 0:
                self.con.wait()
            obj = self.pipe[0]
            self.pipe = self.pipe[1:]
            self.con.notify()
        return obj

class BGCTV(object):
    """Class to control BGCTV"""

    IR_ADDR = ('192.168.1.203', 7001)
    BGCTV_STREAM = 'udp://239.255.42.42:5004'

    MQTT_SERVER = "192.168.1.4"

    BGCTV_CMND_TOPIC = "cmnd/BGCTV/POWER"
    BGCTV_STAT_TOPIC = "stat/BGCTV/POWER"

    PRIO_ANY = 0
    PRIO_MIN = 1
    PRIO_MAX = 2

    OUTPUT_RAW_IMAGE = 0
    OUTPUT_LOGO_CROP = 1

    MODE_KEYFRAME_ANY = 0
    MODE_KEYFRAME_DELTA = 1
    MODE_KEYFRAME_MIXING = 2

    DELTA_THRESHOLD = 768
    STALL_THRESHOLD = 20

    def __init__(self, channelmap, conv):
        self.scale_w, self.scale_h = conv[0]
        w, h, x, y = conv[1]
        self.region = ((x, y, x+w, y+h), (self.scale_w-x-w, y, self.scale_w-x, y+h))
        self.crop_w, self.crop_h = w, h * 2

        self.ir_data = {}

        # UDP socket used to send IR data packet to IR transmitter
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_connected = False
        self.sw_online = False
        self.powered_on = False
        self.current_ch = None

        self.mqtt_client.connect(self.MQTT_SERVER, 1883, 60)
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

        with open(channelmap) as f:
            contents = [ l.strip() for l in f.readlines() if not l.startswith('#') ]
            max_ch = [ int(l.split(':', 2)[0]) for l in contents if l.split(':', 2)[1] == 'max' ][0]
            self.channels = [('_', 'unknown')] * max_ch
            self.scores = np.zeros((max_ch), dtype=np.uint32)
            for l in contents:
                i, ch_id, ch_name = l.split(':', 2)
                i = int(i)
                if i < max_ch:
                    self.channels[i] = (ch_id, ch_name)
            self.labels = max_ch

        self.thread = None

    def ir_send(self, key):
        self.sock.sendto(self.ir_data[key], self.IR_ADDR)

    def on_mqtt_connect(self, mqtt_client, userdata, flags, rc):
        #print("Connected with result code " + str(rc))
        self.mqtt_connected = True

        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        mqtt_client.subscribe(self.BGCTV_STAT_TOPIC)
        mqtt_client.publish(self.BGCTV_CMND_TOPIC, "9")

    def on_mqtt_message(self, mqtt_client, userdata, msg):
        #print(msg.topic + " " + str(msg.payload))

        if msg.topic == self.BGCTV_STAT_TOPIC:
            self.sw_online = True
            if msg.payload == "ON":
                self.powered_on = True
            elif msg.payload == "OFF":
                self.powered_on = False

    def power_on(self):
        self.mqtt_client.publish(self.BGCTV_CMND_TOPIC, "ON")

    def power_off(self):
        self.mqtt_client.publish(self.BGCTV_CMND_TOPIC, "OFF")

    def wait_online(self):
        timeout = 50
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

    def set_channel(self, ch_idx):
        if not self.channels[ch_idx][0].isdigit():
            print("Invalid channel: %d(%s)" % (ch_idx, self.channels[ch_idx][0]))
            return

        print('CH %s: [%s]' % (self.channels[ch_idx][0], self.channels[ch_idx][1]))
        self.ir_send('quit')
        time.sleep(1)
        for num in str(self.channels[ch_idx][0]):
            self.ir_send(num)
            time.sleep(.5)
        self.ir_send('enter')
        self.current_ch = ch_idx

    def next_channel_by_score(self, prio, rand_range=1):
        valid = sorted([ (i, self.scores[i]) for i, x in enumerate(self.channels) if x[0].isdigit() ], key=lambda d: d[1])
        if rand_range > len(valid):
            rand_range = len(valid)

        low = (len(valid) - rand_range) if prio == self.PRIO_MAX else 0
        high = (rand_range - 1) if prio == self.PRIO_MIN else (len(valid) - 1)

        idx = valid[random.randint(low, high)][0]
        self.set_channel(idx)
        return idx, self.scores[idx]

    def score(self, ch_idx, score):
        self.scores[ch_idx] += score

    def capture_raw(self, mode, max_stall, max_skip):
        prev_crop = np.array([0], dtype=np.uint8)
        stalled = 0
        skipped = 0
        container = av.open(self.BGCTV_STREAM)
        for frame in container.decode(video=0):
            if self.stop:
                break

            if not frame.key_frame:
                continue

            #im = frame.to_image().resize(self.resize, Image.BILINEAR)
            # seems like frame.reformat is much more efficient
            im = frame.reformat(self.scale_w, self.scale_h).to_image()

            self.ir_send('quit')

            if mode == self.MODE_KEYFRAME_DELTA or max_stall > 0 or max_skip > 0:
                crop = np.append(np.asarray(im.crop(self.region[0]), dtype=np.uint8), np.asarray(im.crop(self.region[1]), dtype=np.uint8), axis=0)
                d = (np.sum(np.square(crop.astype('float32') - prev_crop.astype('float32'))) / crop.size).astype('uint32')
                if d >= self.DELTA_THRESHOLD:
                    prev_crop = crop
                    stalled = 0
                    skipped = 0
                elif d < self.STALL_THRESHOLD:
                    stalled += 1
                    skipped += 1
                else:
                    stalled = 0
                    skipped += 1
                if max_stall > 0 and stalled == max_stall or max_skip > 0 and skipped == max_skip:
                    break
                if mode == self.MODE_KEYFRAME_DELTA and d < self.DELTA_THRESHOLD:
                    continue

            self.buf.add(im, False)
            im = None
        self.buf.add(None)

    def capture_logo(self, batch_size, mode, max_stall, max_skip):
        prev_crop = np.array([0], dtype=np.uint8)
        mix_sum = np.zeros((self.crop_w, self.crop_h, 3), dtype=np.uint32)
        mix_num = 0
        stalled = 0
        skipped = 0
        batch = None
        num = 0
        container = av.open(self.BGCTV_STREAM)
        for frame in container.decode(video=0):
            if self.stop:
                break

            if not frame.key_frame and mode != self.MODE_KEYFRAME_MIXING:
                continue

            #im = frame.to_image().resize(self.resize, Image.BILINEAR)
            # seems like frame.reformat is much more efficient
            im = frame.reformat(self.scale_w, self.scale_h).to_image()
            crop = np.append(np.asarray(im.crop(self.region[0]), dtype=np.uint8), np.asarray(im.crop(self.region[1]), dtype=np.uint8), axis=0)

            if mode == self.MODE_KEYFRAME_MIXING:
                mix_sum += crop
                mix_num += 1

            if not frame.key_frame:
                continue

            self.ir_send('quit')

            if mode == self.MODE_KEYFRAME_DELTA or max_stall > 0 or max_skip > 0:
                d = (np.sum(np.square(crop.astype('float32') - prev_crop.astype('float32'))) / crop.size).astype('uint32')
                if d > self.DELTA_THRESHOLD:
                    prev_crop = crop
                    stalled = 0
                    skipped = 0
                elif d < self.STALL_THRESHOLD:
                    stalled += 1
                    skipped += 1
                else:
                    stalled = 0
                    skipped += 1
                if max_stall > 0 and stalled == max_stall or max_skip > 0 and skipped == max_skip:
                    break
                if mode == self.MODE_KEYFRAME_DELTA and d < self.DELTA_THRESHOLD:
                    continue

            if mode == self.MODE_KEYFRAME_MIXING:
                crop = (mix_sum / mix_num).astype('uint8')
                mix_sum.fill(0)
                mix_num = 0

            if batch is None:
                batch = np.empty((batch_size, mix_sum.size), dtype=np.uint8)
                num = 0

            batch[num, :] = crop.reshape(1, -1)
            num += 1
            if num == batch_size:
                self.buf.add(batch, False)
                batch = None
        self.buf.add(None)

    def batch_start(self, batch_size, output, mode=MODE_KEYFRAME_ANY, max_stall=0, max_skip=0):
        assert self.thread is None
        self.buf = LogoBuffer(3)
        self.stop = False
        if output == self.OUTPUT_RAW_IMAGE:
            self.thread = threading.Thread(target=self.capture_raw, args=(mode, max_stall, max_skip))
        else:
            self.thread = threading.Thread(target=self.capture_logo, args=(batch_size, mode, max_stall, max_skip))
        self.thread.start()

    def batch_stop(self):
        if (self.thread is not None) and self.thread.isAlive():
            self.stop = True
            while self.buf.get(False) is not None:
                pass
            try:
                self.thread.join()
            except:
                pass
        self.thread = None
        self.buf = None

    def batch_get(self):
        if self.thread is None:
            return (None, None)

        if not self.thread.isAlive():
            self.thread = None
            self.buf = None
            return (None, None)

        bx = self.buf.get()
        if bx is None:
            return (None, None)
        elif type(bx) == Image.Image:
            n = 1
        else:
            n = bx.shape[0]

        if self.current_ch is not None:
            by = np.zeros((n, self.labels), dtype=np.float32)
            by[:, self.current_ch] = 1.0
        else:
            by = None

        return (bx, by)


def main():

    basedir = '.'
    channelmap = join(basedir, 'channels.txt')

    network = ((160, 160, 3), ((32, 32), (64, 64), (128, 128)), (1024,), 1000)
    #imgconv = ((640, 360), (((20, 4, 180, 84),),((460, 4, 620, 84),)))
    imgconv = ((640, 360), (160, 80, 20, 4))
    tv = BGCTV(channelmap, imgconv)

    success = tv.wait_online()
    if not success:
        print('BGCTV Power Control not online, exit.')
        return 2

    #tv.set_channel(42)
    sys.path.append(join(basedir, 'model'))
    import tvlogo

    logo = tvlogo.tvlogo(basedir)

    tv.batch_start(1, tv.OUTPUT_LOGO_CROP, tv.MODE_KEYFRAME_MIXING)

    try:
        while True:
            bx, _ = tv.batch_get()
            if bx is None:
                break

            result = logo.classify(bx)
            bx = None
            print(result)
    except BaseException:
        tv.batch_stop()

if __name__ == "__main__":
    main()
