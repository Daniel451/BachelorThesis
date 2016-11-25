__author__ = 'daniel'

import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import time
import threading
import random


global my_data

path = "/home/daniel/bt-img/test1/"
img = ["ball-1.jpg", "ball-2.jpg", "ball-3.jpg", "ball-4.jpg", "ball-5.jpg"]

my_data = ndimage.imread(path + random.choice(img))


class ComputeThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.nr = 1
        self.i = 0

    def run(self):
        global my_data

        while(self.i < 10):
            self.i += 1
            read_file = random.choice(img)
            print("Compute iteration: " + str(self.i) + " | reading file: " + read_file)
            my_data = ndimage.imread(path + read_file)
            print("type: {0} | shape: {1}".format(type(my_data), my_data.shape))
            time.sleep(1.0)


class Graphic():

    def __init__(self):
        self.i = 0
        self.run()

    def run(self):
        global my_data

        f, axarr = plt.subplots(2, 2)

        self.do_plot(axarr)

        plt.pause(0.01)

        while self.i < 10:
            self.i += 1
            print("Graphic iteration: " + str(self.i))
            self.do_plot(axarr)
            plt.draw()
            plt.pause(0.01)
            time.sleep(1.0)

        plt.close()

    def do_plot(self, axarr):
        axarr[0, 0].imshow(my_data, interpolation="none")

        axarr[0, 1].imshow(my_data[:, :, 0], interpolation="none")

        axarr[1, 0].imshow(my_data[:, :, 1], interpolation="none")

        axarr[1, 1].imshow(my_data[:, :, 2], interpolation="none")


CT = ComputeThread()
CT.start()

G = Graphic()

print("done!")