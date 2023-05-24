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


class Othello():
    def __init__(self, color='b'):
        # define the initial state of the game
        chance = np.ones((8, 8), dtype=int)
        
        white_board = np.zeros((8, 8), dtype=int)
        white_board[3,3], white_board[4,4] = 1,1
        
        black_board = np.zeros((8, 8), dtype=int)
        black_board[3,4], black_board[4,3] = 1,1
        
        self.state = np.array([white_board, black_board, chance])
        self.wcount = 2
        self.bcount = 2
        
    def viz(self):
        # game visulaization

        va = set(self.validAction())

        viz_state = np.zeros((8, 8), dtype=int)

        for i in range(len(self.state[0])):
            for j in range(len(self.state[0][0])):
                if self.state[0][i][j] == 1:
                    viz_state[i][j] = 1
                if self.state[1][i][j] == 1:
                    viz_state[i][j] = -1

        fig, ax = plt.subplots()
        ax.set_facecolor('green')

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)

        for i in range(8):
            for j in range(8):
                if viz_state[i, j] == 1:
                    circle = plt.Circle((j + 0.5, i + 0.5), 0.4, color='white', ec='black')
                    ax.add_artist(circle)
                elif viz_state[i, j] == -1:
                    circle = plt.Circle((j + 0.5, i + 0.5), 0.4, color='black', ec='black')
                    ax.add_artist(circle)
                if (i, j) in va:
                    circle = plt.Circle((j + 0.5, i + 0.5), 0.4, color='none', ec='blue', linestyle='dashed')
                    ax.add_artist(circle)

        ax.set_xticks(np.arange(0, 8, 1))
        ax.set_yticks(np.arange(0, 8, 1))
        ax.grid(color='black', linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        # RETURN IMG


        def get_img_from_fig(fig, dpi=180):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img_np = np.array(img)
            return img
        return get_img_from_fig(fig)
        
    def validAction(self, state=[]):
        # returns list of valid actions in the current game state

        if len(state) == 0:
          state = self.state

        player = state[2][0][0]
        opponent = player ^ 1
        valid_moves = []
        for i in range(8):
            for j in range(8):
                if state[0][i][j]!=0 or state[1][i][j]!=0: continue
                for di, dj in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:  
                    x, y = i+di, j+dj
                    if 0<=x<8 and 0<=y<8 and state[opponent][x][y] == 1:  
                        while 0<=x<8 and 0<=y<8 and state[opponent][x][y] == 1:  
                            x, y = x+di, y+dj
                        if 0<=x<8 and 0<=y<8 and state[player][x][y] == 1: 
                            valid_moves.append((i,j))
        return valid_moves

    def applyAction(self, action):
        # applies the 'action' to the current game state

        i, j = action
        player = self.state[2][0][0]
        opponent = player ^ 1
        if i==-1:     # incase of no valid action
            self.state[2] = np.ones((8,8), dtype=int) * opponent
            return

        self.state[player][i][j] = 1

        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            x, y = i + di, j + dj
            flipped_pieces = []
            while 0 <= x < 8 and 0 <= y < 8 and self.state[opponent][x][y] == 1:
                flipped_pieces.append((x, y))
                x, y = x + di, y + dj
            if 0 <= x < 8 and 0 <= y < 8 and self.state[player][x][y] == 1 and len(flipped_pieces) > 0:
                for x, y in flipped_pieces:
                    self.state[player][x][y] = 1
                    self.state[opponent][x][y] = 0
                    if player==1:
                        self.bcount += 1
                        self.wcount -= 1
                    else:
                        self.wcount += 1
                        self.bcount -= 1
        if player==1:
            self.bcount += 1
        else:
            self.wcount += 1
                    
        self.state[2] = np.ones((8, 8), dtype=int) * opponent
    
    def getWinner(self):
        # returns winner at terminal state; 1 if white, -1 if black, 0 otherwise

        white = np.count_nonzero(self.state[0]==1)
        black = np.count_nonzero(self.state[1]==1)
        if white>black: return 1
        elif black>white: return -1
        return 0