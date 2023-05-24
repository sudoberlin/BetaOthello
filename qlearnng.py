from othello import Othello
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

class QLearning(Othello):
    def __init__(self, network, target, epsilon=1.0, gamma=0.95):
        self.network = network
        self.target = target
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer = deque()


    def policy(self, state, inference=False):
    # epsilon greedy behaviour policy
        rand = random.random()
        valid_actions = self.validAction(state)
        if rand < self.epsilon and not inference:  # returns random action
            if len(valid_actions) == 0:
                return (-1, -1)
            return valid_actions[random.randint(0, len(valid_actions) - 1)]
        argmax = (-1, -1)
        max_ = -10**9
        for a in valid_actions:
            one_hot_action = np.zeros((1, 8, 8))
            one_hot_action[0][a[0]][a[1]] = 1
            state_action = np.vstack((state, one_hot_action))
            # Add a new axis for the batch size
            state_action = np.expand_dims(state_action, axis=0)
            q_val = self.network(state_action)
            if q_val > max_:
                max_ = q_val
                argmax = a
        return argmax  # returns action with max q-value




    def collect_experience(self,episodes=100):
        # collects experience in buffer
        for ep in tqdm.tqdm(range(episodes)):
            states, actions = [], []
            game = Othello()
            invalid = False
            while game.wcount!=0 and game.bcount!=0 and game.bcount+game.wcount<64:
                action = self.policy(game.state)
                if action[0]==-1 and invalid: break
                elif action[0]==-1: invalid=True
                else: invalid=False
                states.append(copy.copy(game.state))
                actions.append(action)
                game.applyAction(action)
            #winner = game.getWinner() # 1 if white, -1 if black
            for t in range(len(states)-1):
                if t % 2 == 0:
                  reward = game.bcount/(game.wcount + 0.1) - 1
                elif t % 2 != 0:
                  reward = game.wcount/(game.bcount + 0.1) - 1
                if actions[t][0]!=-1:
                    self.buffer.append((states[t], actions[t], reward, states[t+1]))
                if len(self.buffer) > 1000:
                    self.buffer.popleft()


    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 64)  # random action
        q_values = self.model.predict(state)
        return np.argmax(q_values)
