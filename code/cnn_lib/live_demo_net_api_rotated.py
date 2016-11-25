import cv2
import numpy as np
import time
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
sys.path.append(str(os.environ["BTEX"]))
from extra.img_ops import ImageOperations as ImgOps
from extra.array_type_printer import ArrayType as AT 

def net_api(net, img):
    img = img[0][:, :, 0:3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_net = ImgOps.rescale_values(img, upper_bound=1.0, dtype=np.float32)
    
    print("start processing...")
    time_start = time.time()
    nn_out_x, nn_out_y, nn_out_x_softmax, nn_out_y_softmax, top11_x, top11_y = net.run_network([img_net])
    print("finished processing! Execution time: {:.4f}".format(time.time()-time_start))

    # unpack values
    nn_out_x = nn_out_x[0]
    nn_out_y = nn_out_y[0]
    nn_out_x_softmax = nn_out_x_softmax[0]
    nn_out_y_softmax = nn_out_y_softmax[0]

    # generate heatmap for (x, y)
    heatmap = np.zeros((600, 800))
    heatmap += nn_out_x_softmax[None, :]
    heatmap += nn_out_y_softmax[:, None]
    heatmap *= 0.5

    heatmap_linear = np.zeros((600, 800))
    heatmap_linear += nn_out_x[None, :]
    heatmap_linear += nn_out_y[:, None]
    heatmap_linear *= 0.5

    # plot configuration
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    p_img = plt.subplot(gs[0, 0]) 
    p_y = plt.subplot(gs[0, 1]) 
    p_x = plt.subplot(gs[1, 0]) 
    p_heatmap = plt.subplot(gs[1, 1]) 

    pl_img = plt.subplot(gs[0, 2]) 
    pl_y = plt.subplot(gs[0, 3]) 
    pl_x = plt.subplot(gs[1, 2]) 
    pl_heatmap = plt.subplot(gs[1, 3]) 

    # softmax
    p_img.imshow(img, interpolation="None")
    p_img.set_title("input test image")

    p_heatmap.imshow(img, interpolation="None")
    p_heatmap.imshow(heatmap, interpolation="None", alpha=0.35)
    p_heatmap.set_title("combined heatmap | softmax output")

    p_x.plot(nn_out_x_softmax, color="magenta", marker="x")
    p_x.set_title("x-axis | softmax output | x={}".format(np.argmax(nn_out_x_softmax)))
    p_y.plot(nn_out_y_softmax, np.arange(nn_out_y_softmax.size), color="magenta", marker="x")
    p_y.invert_xaxis()
    p_y.invert_yaxis()
    p_y.set_title("y-axis | softmax output | y={}".format(np.argmax(nn_out_y_softmax)))


    # linear
    pl_img.imshow(img, interpolation="None")
    pl_img.set_title("input test image")

    pl_heatmap.imshow(img, interpolation="None")
    pl_heatmap.imshow(heatmap_linear, interpolation="None", alpha=0.35)
    pl_heatmap.set_title("combined heatmap | linear output")

    pl_x.plot(nn_out_x, color="magenta", marker="x")
    pl_x.set_title("x-axis | linear output | x={}".format(np.argmax(nn_out_x)))
    pl_y.plot(nn_out_y, np.arange(nn_out_y.size), color="magenta", marker="x")
    pl_y.invert_xaxis()
    pl_y.invert_yaxis()
    pl_y.set_title("y-axis | linear output | y={}".format(np.argmax(nn_out_y)))

    plt.show()


