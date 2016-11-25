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

# network model
from cnn_series10.models.cnn_s10_ex01 import Net_s10_ex01 as net_cls 



########################
# start the experiment #
########################
net = net_cls("cnn_exp10_01")
net.load_network()


def net_api(img):
    img = ImgOps.rescale_values(img, upper_bound=255.0, dtype=np.uint8)
    print("oimg", AT.get_type(img))
    img_net = ImgOps.rescale_values(img, upper_bound=1.0, dtype=np.float32)
    print("img_net", AT.get_type(img_net))

    print("start processing...")
    time_start = time.time()
    nn_out_x, nn_out_y, nn_out_x_softmax, nn_out_y_softmax, top11_x, top11_y = net.run_network(img_net)
    print("finished processing! Execution time: {:.4f}".format(time.time()-time_start))

    print("net output x", AT.get_type(nn_out_x))
    print("net output y", AT.get_type(nn_out_y))

    print("net output softmax x", AT.get_type(nn_out_x_softmax))
    print("net output softmax y", AT.get_type(nn_out_x_softmax))

    img_grey = cv2.cvtColor(img[-1], cv2.COLOR_BGR2GRAY)

    # generate heatmap for (x, y)
    heatmap = np.zeros((600, 800))
    heatmap += nn_out_x_softmax[0][None, :]
    heatmap += nn_out_y_softmax[0][:, None]
    heatmap *= 0.5
    heatmap *= 255.0 / np.max(heatmap) 
    heatmap = heatmap.clip(0.0, 255.0)
    heatmap = heatmap.astype(np.uint8)

    print("heatmap", AT.get_type(heatmap))

    out_heatmap = cv2.addWeighted(img_grey, 0.5, heatmap, 0.8, 1.0) 
    out_heatmap = cv2.applyColorMap(out_heatmap, cv2.COLORMAP_JET)

    print("plot image", AT.get_type(out_heatmap))

    cv2.imshow("network output", out_heatmap)
    cv2.waitKey(0)

for img_str in sys.argv[1:]:
    net_api([mpimg.imread(img_str)])

print("exiting...")



