from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
import sys
import numpy as np
import h5py

sys.path.append(str(os.environ["BTEX"]))
from extra.array_type_printer import ArrayType as AT


# config 
experiment1 = "cnn_exp11_17/"
experiment2 = "cnn_exp11_22/"
image_width = 200
image_height = 150

logpath = os.environ["BTLOG"]


#########################
### experiment 1 data ###
#########################
h5f_exp1 = h5py.File(logpath + experiment1 + "training_log_data.h5", "r")

print("\n--- experiment 1---")
print("> " + h5f_exp1.attrs["file_name"] + " " + h5f_exp1.attrs["file_time"])
print("|-> " + h5f_exp1.attrs["creator"] + " (script)")
print("|-> " + h5f_exp1.attrs["HDF5_version"] + " (HDF5 version)")
print("|-> " + h5f_exp1.attrs["h5py_version"] + " (h5py version)")

ex1_num_iterations = h5f_exp1.attrs["amount_of_train_steps"]
ex1_log_every_n_steps = h5f_exp1.attrs["log_every_n_steps"]
interval1 = np.arange(ex1_log_every_n_steps, ex1_num_iterations + ex1_log_every_n_steps, ex1_log_every_n_steps)


# get data out of the h5f_exp1 object
ex1_accu_x_mean = h5f_exp1["testdata"]["accu_x_mean"][:] / float(image_width)
ex1_accu_y_mean = h5f_exp1["testdata"]["accu_y_mean"][:] / float(image_height)

ex1_top11_x_mean = h5f_exp1["testdata"]["top11_x_mean"][:]
ex1_top11_y_mean = h5f_exp1["testdata"]["top11_y_mean"][:]

ex1_ce_x_mean = h5f_exp1["testdata"]["ce_x_mean"][:]
ex1_ce_y_mean = h5f_exp1["testdata"]["ce_y_mean"][:]


#########################
### experiment 2 data ###
#########################
h5f_exp2 = h5py.File(logpath + experiment2 + "training_log_data.h5", "r")
print("\n--- experiment 1---")
print("> " + h5f_exp2.attrs["file_name"] + " " + h5f_exp2.attrs["file_time"])
print("|-> " + h5f_exp2.attrs["creator"] + " (script)")
print("|-> " + h5f_exp2.attrs["HDF5_version"] + " (HDF5 version)")
print("|-> " + h5f_exp2.attrs["h5py_version"] + " (h5py version)")

ex2_num_iterations = h5f_exp2.attrs["amount_of_train_steps"]
ex2_log_every_n_steps = h5f_exp2.attrs["log_every_n_steps"]
interval2 = np.arange(ex2_log_every_n_steps, ex2_num_iterations + ex2_log_every_n_steps, ex2_log_every_n_steps)


# get data out of the h5f_exp1 object
ex2_accu_x_mean = h5f_exp2["testdata"]["accu_x_mean"][:] / float(image_width)
ex2_accu_y_mean = h5f_exp2["testdata"]["accu_y_mean"][:] / float(image_height)

ex2_top11_x_mean = h5f_exp2["testdata"]["top11_x_mean"][:]
ex2_top11_y_mean = h5f_exp2["testdata"]["top11_y_mean"][:]

ex2_ce_x_mean = h5f_exp2["testdata"]["ce_x_mean"][:]
ex2_ce_y_mean = h5f_exp2["testdata"]["ce_y_mean"][:]




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
    subplot.set_xlabel("iterations", fontsize=25)
    subplot.set_ylabel("values", fontsize=25)
    subplot.set_title(ptitle, fontsize=30)

    # legend
    subplot.legend()
    subplot.legend(prop={"size": 14}, loc=2)


plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05,
                    hspace=0.25, wspace=0.4)

gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

top11_x = plt.subplot(gs[0, 0])
accu_x = plt.subplot(gs[0, 1])
ce_x = plt.subplot(gs[0, 2])

top11_y = plt.subplot(gs[1, 0])
accu_y = plt.subplot(gs[1, 1])
ce_y = plt.subplot(gs[1, 2])

##############
### x data ###
##############
top11_x.plot(interval1, ex1_top11_x_mean, linestyle="solid", marker="None", color="blue", label=experiment1[:-1])
top11_x.plot(interval2, ex2_top11_x_mean, linestyle="solid", marker="None", color="red", label=experiment2[:-1])
apply_plot_style(top11_x, "x: top11 accuracy")

accu_x.plot(interval1, ex1_accu_x_mean, linestyle="solid", marker="None", color="blue", label=experiment1[:-1])
accu_x.plot(interval2, ex2_accu_x_mean, linestyle="solid", marker="None", color="red", label=experiment2[:-1])
apply_plot_style(accu_x, "x: abs accuracy error rate")

ce_x.plot(interval1, ex1_ce_x_mean, linestyle="solid", marker="None", color="blue", label=experiment1[:-1])
ce_x.plot(interval2, ex2_ce_y_mean, linestyle="solid", marker="None", color="red", label=experiment2[:-1])
apply_plot_style(ce_x, "x: cross entropy")


##############
### y data ###
##############
top11_y.plot(interval1, ex1_top11_y_mean, linestyle="solid", marker="None", color="blue", label=experiment1[:-1])
top11_y.plot(interval2, ex2_top11_y_mean, linestyle="solid", marker="None", color="red", label=experiment2[:-1])
apply_plot_style(top11_y, "y: top11 accuracy")

accu_y.plot(interval1, ex1_accu_y_mean, linestyle="solid", marker="None", color="blue", label=experiment1[:-1])
accu_y.plot(interval2, ex2_accu_y_mean, linestyle="solid", marker="None", color="red", label=experiment2[:-1])
apply_plot_style(accu_y, "y: abs accuracy error rate")

ce_y.plot(interval1, ex1_ce_y_mean, linestyle="solid", marker="None", color="blue", label=experiment1[:-1])
ce_y.plot(interval2, ex2_ce_y_mean, linestyle="solid", marker="None", color="red", label=experiment2[:-1])
apply_plot_style(ce_y, "y: cross entropy")


plt.show()

