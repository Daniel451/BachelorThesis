import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np

img = mpimg.imread("/home/daniel/Dropbox/Studium/computer science BSc/bachelor-thesis/bt-img/test-training-1/train/img_fr_nov15_000.613_grey.png")

distri = np.random.normal(loc=400, scale=20, size=800)

f, axarr = plt.subplots(1, 3)
axarr[0].imshow(img, interpolation="none", cmap="gray")
axarr[1].plot(np.arange(0, 800), distri, linestyle="solid", marker="x", color="black", label="label1")
axarr[1].plot(np.random.rand(800), linestyle="solid", marker="x", color="magenta", label="label2")
axarr[1].legend()
axarr[2].plot(distri, linestyle="solid", marker="x", color="#ad892f")
plt.show()
