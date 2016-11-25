#!/sw/env/gcc-4.9.3_openmpi-1.8.8/pkgsrc/2015Q4/bin/python2
import sys
import os
import numpy as np
import cv2
import time
from matplotlib import image as mpimg
# append the experiments/code directory to python sys path
sys.path.append(str(os.environ["BTEX"]))
from distributions.convert_to_distribution import DistributionConverter as DC
from cnn_lib.CamReader import Recorder
from extra.array_type_printer import ArrayType as AT
from extra.img_ops import ImageOperations as ImgOps
from cnn_lib.img_demo_net_api_rotated import net_api

# network model
from cnn_models.models.cnn_s11_ex02 import NetworkModel as net_cls



########################
# start the experiment #
########################
net = net_cls("cnn_exp11_20")
net.load_network()

for img_str in sys.argv[1:]:
    net_api(net, [mpimg.imread(img_str)])

print("exiting...")



