from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
import sys
import numpy as np
import h5py

sys.path.append(str(os.environ["BTEX"]))
from extra.array_type_printer import ArrayType as AT


# config 
experiment = "cnn_exp11_17/"
image_width = 200
image_height = 150

logpath = os.environ["BTLOG"]


# actual plot data
h5f = h5py.File(logpath + experiment + "training_log_data.h5", "r")

print("> " + h5f.attrs["file_name"] + " " + h5f.attrs["file_time"])
print("|-> " + h5f.attrs["creator"] + " (script)")
print("|-> " + h5f.attrs["HDF5_version"] + " (HDF5 version)")
print("|-> " + h5f.attrs["h5py_version"] + " (h5py version)\n")

num_iterations = h5f.attrs["amount_of_train_steps"]
log_every_n_steps = h5f.attrs["log_every_n_steps"]
interval = np.arange(log_every_n_steps, num_iterations+log_every_n_steps, log_every_n_steps)


# get data out of the h5f object
accu_x_mean = h5f["testdata"]["accu_x_mean"][:] / float(image_width)
accu_y_mean = h5f["testdata"]["accu_y_mean"][:] / float(image_height)
accu_x_median = h5f["testdata"]["accu_x_median"][:] / float(image_width)
accu_y_median = h5f["testdata"]["accu_y_median"][:] / float(image_height)

top11_x_mean = h5f["testdata"]["top11_x_mean"][:]
top11_y_mean = h5f["testdata"]["top11_y_mean"][:]
top11_x_median = h5f["testdata"]["top11_x_median"][:]
top11_y_median = h5f["testdata"]["top11_y_median"][:]

ce_x_mean = h5f["testdata"]["ce_x_mean"][:]
ce_y_mean = h5f["testdata"]["ce_y_mean"][:]
ce_x_median = h5f["testdata"]["ce_x_median"][:]
ce_y_median = h5f["testdata"]["ce_y_median"][:]

print("accuracy x/y mean/median")
AT.print_type(accu_x_mean)
AT.print_type(accu_y_mean)
AT.print_type(accu_x_median)
AT.print_type(accu_y_median)

print("")

print("top11 x/y mean/median")
AT.print_type(top11_x_mean)
AT.print_type(top11_y_mean)
AT.print_type(top11_x_median)
AT.print_type(top11_y_median)

print("")

print("cross entropy x/y mean/median")
AT.print_type(ce_x_mean)
AT.print_type(ce_y_mean)
AT.print_type(ce_x_median)
AT.print_type(ce_y_median)

def apply_plot_style(subplot, ptitle):
    # ticks
    subplot.tick_params(labelsize=18)
    subplot.tick_params(labelright=True)
    subplot.tick_params(which="both", width=2, color="black")
    subplot.tick_params(which="major", length=15)
    subplot.tick_params(which="minor", length=5)
    subplot.minorticks_on()

    subplot.grid(True)

    # labels and title
    subplot.set_xlabel("iterations", fontsize=35)
    subplot.set_ylabel("values", fontsize=35)
    subplot.set_title(ptitle, fontsize=40)

    # legend
    subplot.legend()
    subplot.legend(prop={"size": 18}, loc=2)

# create the figure
f = plt.figure()

plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05,
                    hspace=0.1, wspace=0.4)

gs = gridspec.GridSpec(1, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
top11 = plt.subplot(gs[0, 0])
accu = plt.subplot(gs[0, 1])
ce = plt.subplot(gs[0, 2])


top11.plot(interval, top11_x_mean, linestyle="solid", marker="None", color="blue", label="top11 x (mean)")
top11.plot(interval, top11_y_mean, linestyle="solid", marker="None", color="red", label="top11 y (mean)")
#top11.plot(interval, top11_x_median, linestyle="solid", marker="None", color="green", label="top11 x (median)")
#top11.plot(interval, top11_y_median, linestyle="solid", marker="None", color="magenta", label="top11 y (median)")
apply_plot_style(top11, "top11 accuracy")

accu.plot(interval, accu_x_mean, linestyle="solid", marker="None", color="blue", label="accuracy x (mean)")
accu.plot(interval, accu_y_mean, linestyle="solid", marker="None", color="red", label="accuracy y (mean)")
apply_plot_style(accu, "abs accuracy error rate")

ce.plot(interval, ce_x_mean, linestyle="solid", marker="None", color="blue", label="cross-entropy x (mean)")
ce.plot(interval, ce_y_mean, linestyle="solid", marker="None", color="red", label="cross-entropy y (mean)")
apply_plot_style(ce, "cross entropy error")


plt.show()
#f.set_size_inches(12.7, 12.0)
#f.savefig("/home/daniel/tmp/error_rate_old_dataset.pdf", dpi=80)

