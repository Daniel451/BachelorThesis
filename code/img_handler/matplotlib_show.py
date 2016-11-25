import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import time
import threading
import random

__author__ = 'daniel'



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


class Graphic:

    def __init__(self):
        self.i = 0
        self.run()

    def run(self):
        global my_data

        plt.imshow(my_data, interpolation="none")
        plt.pause(0.01)

        while self.i < 10:
            self.i += 1
            print("Graphic iteration: " + str(self.i))
            plt.imshow(my_data, interpolation="none")
            plt.draw()
            plt.pause(0.01)
            time.sleep(1.0)

        plt.close()


CT = ComputeThread()
CT.start()

G = Graphic()

print("done!")