# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:18:01 2022

@author: papad
"""
import torch

buffer = []

board = torch.zeros(3,3)
next_board = torch.rand(3,3)
reward = 1
action = (1,2)

sequence = {'state': board, 'next_state': next_board, 'reward':reward, 'action':action}

buffer.append(sequence)

print(buffer[0]['state'])

