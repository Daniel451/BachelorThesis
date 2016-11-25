import sys


if len(sys.argv) != 4:
    raise Exception("parameters do not match\n\nscheme: mirror_labels.py /path/to/labels.txt image_width image_height\n"\
                    +"\nfull example: mirror_labels.py /home/some_user/labels.txt 800 600\n")

filepath = str(sys.argv[1])
width = int(sys.argv[2])
height = int(sys.argv[3])

# initialize variables
label_buffer = []

# read in labels file
with open(filepath, "r") as f_labels:  # read file containing labels
    label_buffer = f_labels.readlines()  # read in all lines

# filter all lines not containing "===" <-- separator for img label data
label_buffer = filter(lambda e: "===" in e, label_buffer)

# remove linebreaks and potential whitespaces
label_buffer = [e.strip() for e in label_buffer]

label_buffer = sorted(label_buffer)

# iterate over all lines
for key, e in enumerate(label_buffer):
    # seperate img filename and xy label data
    fname, xydata = e.split("===")

    # seperate x and y label data
    x, y = xydata.split(",")

    # caution: first char in x/y is "x"/"y"!
    # original encoding looks like this: x123,y456
    xint, yint = int(x[1:]), int(y[1:])

    # mirror x data
    diff = int(width / 2.0) - xint
    xmirror = xint + 2*diff 

    label_buffer[key] = label_buffer[key].replace("x"+str(xint), "x"+str(xmirror)) + "\n"

# write new data
with open(filepath[:-4] + "-mirrored.txt", "w") as f_mirrored_labels:
    f_mirrored_labels.writelines(label_buffer)


