import library
from othello import Othello
from bnn import *
from qlearnng import QLearning

from keras import models
from keras.models import load_model

import sys
from os import linesep
from numpy.core.function_base import linspace
import copy
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
from keras import layers
import tensorflow_probability as tfp
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import cv2
import io
import tqdm
from collections import deque
import os
from keras import models
from keras.models import load_model


input_shape = (4,8,8,) # (State shape flattened + Action shape)""" #should be either [(8,8,3), (8,8)] or [8,8,4]


network = build_bnn(input_shape, 1)  # Temporarily use 1 as num_samples
target_network = None  # Set the target network to None as it is not used in this context

#q_learning = QLearning(network, target_network, 0.1)
#q_learning.collect_experience(episodes=10)

#num_samples = 32
batch_size = 32
bnn_model = build_bnn(input_shape, batch_size)
compiled_bnn = compile_bnn(bnn_model)
q_learning = QLearning(bnn_model, target_network, 0.1)


# Train BNN
for i in range(100): #10
    q_learning.collect_experience(episodes=1)
    state_action_pairs, rewards = prepare_data(q_learning.buffer, batch_size)
    #for i in range(len(rewards)):
    train_bnn(compiled_bnn, state_action_pairs, rewards, epochs=10)
# Human VS RL  viz return sttment