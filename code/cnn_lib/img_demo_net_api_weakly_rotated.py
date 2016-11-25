import cv2
import numpy as np
import time
import numpy as np
import os
import sys
import math
from itertools import izip, count, cycle
from matplotlib import pyplot as plt
from matplotlib import gridspec
sys.path.append(str(os.environ["BTEX"]))
from extra.img_ops import ImageOperations as ImgOps
from extra.array_type_printer import ArrayType as AT 

def net_api(net, img):
    img = img[0][:, :, 0:3]
    img_net = ImgOps.rescale_values(img, upper_bound=1.0, dtype=np.float32)
    
    print("start processing...")
    time_start = time.time()
    nn_out, hconv1, hpool1, hconv2, hpool2, hconv3, hpool3 = net.run_network([img_net])
    print("finished processing! Execution time: {:.4f}".format(time.time()-time_start))

    AT.print_type(nn_out)
    AT.print_type(hconv1)
    AT.print_type(hpool1)
    AT.print_type(hconv2)
    AT.print_type(hpool2)
    AT.print_type(hconv3)
    AT.print_type(hpool3)

    # unpack values
    nn_out = nn_out[0]

    def get_next_10th(x):
        if x % 10 != 0:
            return (10 - x % 10) + x
        else:
            return x

    for feature_tensor, layer_name in izip((ft for ft in [hconv1, hpool1, hconv2, hpool2, hconv3, hpool3]),\
                                           ["hconv1", "hpool1", "hconv2", "hpool2", "hconv3", "hpool3"]):

        # plots
        gs = gridspec.GridSpec(2, 2)
        o_img = plt.subplot(gs[0, 0])
        mean_img = plt.subplot(gs[0, 1])
        fmaps = plt.subplot(gs[1, 0:2])

        # plot configuration
        num_feature_maps = feature_tensor.shape[3]
        num_subplots = get_next_10th(num_feature_maps) 
        num_subplot_rows = num_subplots / 10
        fmap_width = feature_tensor.shape[2]
        fmap_height = feature_tensor.shape[1]

        pwidth = (fmap_width+5) * 10 - 5
        pheight = (fmap_height+5) * num_subplot_rows - 5

        img_holder = np.zeros((pheight, pwidth))
        mean_img_holder = np.zeros((fmap_height, fmap_width)).astype(np.float32)

        for i, j, img_i in izip((int(math.floor(g / 10)) for g in xrange(num_subplots)), cycle(xrange(10)), xrange(num_feature_maps)):
            start_row = i * (fmap_height + 5)
            start_column = j * (fmap_width + 5)

            img_holder[start_row : start_row+fmap_height, start_column : start_column+fmap_width] = feature_tensor[0, :, :, img_i] 
            mean_img_holder += feature_tensor[0, :, :, img_i]

        mean_img_holder = mean_img_holder / float(num_feature_maps)

        o_img.imshow(img)
        o_img.set_title("original image")

        #mean_img.imshow(mean_img_holder, cmap="hot")
        mean_img.imshow(mean_img_holder)
        mean_img.set_title("mean image")

        fmaps.imshow(img_holder, cmap="hot")
        #fmaps.colorbar()
        fmaps.set_title("layer: {} | feature map shape: {}".format(layer_name, feature_tensor.shape), fontsize=30)

        plt.show()


