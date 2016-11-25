#!/sw/env/gcc-4.9.3_openmpi-1.8.8/pkgsrc/2015Q4/bin/python2
import sys
import os
sys.path.append(str(os.environ["BTEX"]) + "rnn/")
import tensorflow as tf
from rnn_model import RNN



net = RNN(experiment="rnn_ex005",
          hidden_size=100,
          batch_size=1,
          learning_rate=1e-4,
          log_every_n_steps=1e4,
          activation_function=tf.nn.softsign)

net.build_model()
#net.train(num_iterations=1e6+1)
net.load_network()
net.visual_evaluation()


print("done.")
