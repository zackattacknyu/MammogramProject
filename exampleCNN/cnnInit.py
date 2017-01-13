import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import pickle
import os
import gzip

import numpy as np
# import theano
#
# import lasange
# from lasange import layers
# from lasagne.update import nesterov_momentum
#
# from nolearn.lasange import NeuralNet
# from nolearn.lasange import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




filename = 'mnist.pkl.gz'
with gzip.open(filename, 'rb') as f:
    data = pickle.load(f)