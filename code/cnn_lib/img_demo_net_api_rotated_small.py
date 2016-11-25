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

def net_api(net, img, fname="", savefig=False, i=0):
    img = img[0][:, :, 0:3]
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
    heatmap = np.zeros((150, 200))
    heatmap += nn_out_x_softmax[None, :]
    heatmap += nn_out_y_softmax[:, None]
    heatmap *= 0.5

    heatmap_linear = np.zeros((150, 200))
    heatmap_linear += nn_out_x[None, :]
    heatmap_linear += nn_out_y[:, None]
    heatmap_linear *= 0.5

    # plot configuration
    f = plt.figure()
    gs = gridspec.GridSpec(2, 4)

    border_size = 0.1
    plt.subplots_adjust(left=border_size-0.05, right=1.03-border_size, top=1.0-border_size, bottom=0.05,
                        hspace=0.5, wspace=0.5)

    p_img = plt.subplot(gs[0, 0]) 
    p_y = plt.subplot(gs[0, 1], sharey=p_img) 
    p_x = plt.subplot(gs[1, 0], sharex=p_img) 
    p_heatmap = plt.subplot(gs[1, 1]) 

    pl_img = plt.subplot(gs[0, 2]) 
    pl_y = plt.subplot(gs[0, 3], sharey=pl_img) 
    pl_x = plt.subplot(gs[1, 2], sharex=pl_img) 
    pl_heatmap = plt.subplot(gs[1, 3]) 

    # font size
    plot_font_size = 12

    # softmax
    p_img.imshow(img, interpolation="None")
    p_img.set_xlim([0, 199])
    p_img.set_ylim([0, 150])
    p_img.set_title("input test image", fontsize=plot_font_size)

    p_heatmap.imshow(img, interpolation="None")
    p_heatmap.imshow(heatmap, interpolation="None", alpha=0.35)
    p_heatmap.set_title("combined heatmap\nsoftmax output", fontsize=plot_font_size)

    p_x.plot(nn_out_x_softmax, color="magenta", marker="x")
    p_x.set_title("x-axis\nsoftmax output | x={}".format(np.argmax(nn_out_x_softmax)), fontsize=plot_font_size)
    p_y.plot(nn_out_y_softmax, np.arange(nn_out_y_softmax.size), color="magenta", marker="x")
    p_y.set_ylim([0, 150])
    p_y.invert_xaxis()
    p_y.invert_yaxis()
    p_y.set_title("y-axis\nsoftmax output | y={}".format(np.argmax(nn_out_y_softmax)), fontsize=plot_font_size)


    # linear
    pl_img.imshow(img, interpolation="None")
    pl_img.set_xlim([0, 199])
    pl_img.set_ylim([0, 150])
    pl_img.set_title("input test image", fontsize=plot_font_size)

    pl_heatmap.imshow(img, interpolation="None")
    pl_heatmap.imshow(heatmap_linear, interpolation="None", alpha=0.35)
    pl_heatmap.set_title("combined heatmap\nlinear output", fontsize=plot_font_size)

    pl_x.plot((-1, 201), (0, 0), color="black", linestyle="solid")
    pl_x.plot(nn_out_x, color="magenta", marker="x")
    pl_x.set_title("x-axis\nlinear output | x={}".format(np.argmax(nn_out_x)), fontsize=plot_font_size)

    pl_y.plot((0, 0), (-1, 151), color="black", linestyle="solid")
    pl_y.plot(nn_out_y, np.arange(nn_out_y.size), color="magenta", marker="x")
    pl_y.set_ylim([0, 150])
    pl_y.invert_xaxis()
    pl_y.invert_yaxis()
    pl_y.set_title("y-axis\nlinear output | y={}".format(np.argmax(nn_out_y)), fontsize=plot_font_size)

    if savefig:
        f.set_size_inches(14.0, 5.0)
        fname = fname.split("/")[-1]
        f.savefig("/home/daniel/tmp/processed_sequences/{:0>3}.png".format(i), dpi=120)
    else:
        plt.show()


