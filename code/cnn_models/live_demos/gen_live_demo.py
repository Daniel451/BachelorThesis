#!/sw/env/gcc-4.9.3_openmpi-1.8.8/pkgsrc/2015Q4/bin/python2
import sys
import os
import numpy as np
import cv2
from matplotlib import image as mpimg
import time

# append the experiments/code directory to python sys path
sys.path.append(str(os.environ["BTEX"]))

from distributions.convert_to_distribution import DistributionConverter as DC
from cnn_lib.CamReader import Recorder
from cnn_lib.live_demo_net_api import net_api
from extra.array_type_printer import ArrayType as AT
from extra.img_ops import ImageOperations as ImgOps

# network model to load
from cnn_series10.models.cnn_s10_ex01 import Net_s10_ex01 as net_cls 


###################
# start live demo #
###################
net = net_cls("cnn_exp10_01")
net.load_network()

R = Recorder(camera=1, width=800, height=600)
R.capture(net, net_api)

print("exiting...")


