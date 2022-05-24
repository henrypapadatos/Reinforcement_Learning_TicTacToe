# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:22:49 2022

@author: papad
"""

import numpy as np

    
def is_feasible(board):
    
    if abs(board.count(-1)-board.count(1)) > 1:       
        return False
    
    else:
        return True

def is_already_rotated(States, board):
    array = np.reshape(board, (3,3))
    
    for _ in range(3):
        
        array = np.rot90(array)
        
        board = np.reshape(array, 9).tolist()
        
        if board in States:
            return True
    
    return False

def is_already_reflected(States, board):
    array = np.reshape(board, (3,3))
    
    comparison = np.flipud(array)
    board = np.reshape(comparison, 9).tolist()
    if board in States:
        return True
    
    comparison = np.fliplr(array)
    board = np.reshape(comparison, 9).tolist()
    if board in States:
        return True
    
    comparison = np.transpose(array)
    board = np.reshape(comparison, 9).tolist()
    if board in States:
        return True
    
    comparison = np.transpose(np.flipud(array))
    board = np.reshape(comparison, 9).tolist()
    if board in States:
        return True
    
    return False 

def define_states_graph():
    
    # sym = [0, 0 , 1, 0 , 0 ,-1, 0 , 0, 0]
    # test = [1,0, 0, -1, 0, 0, 0, 0, 0]
    # listt = []
    # listt.append(test)
    
          
    States = []
    
    board = [0] *9
    
    
    for unique_id in range(pow(3,9)):
        tile_val = unique_id
        for x in range(9):
            board[x] = tile_val%3
            tile_val//=3
            
        while (2 in board):
            board[board.index(2)] = -1
            
        # if board == sym:
        #     pass
            
        if (is_feasible(board) == True):
            # and is_already_rotated(States, board) == False ):
            # and is_already_reflected(States, board) == False):
            #States.append(np.reshape(board, (3,3)))
            States.append(board.copy())
            
            # print(np.reshape(board, (3,3)))
    return States

def initialize_Qvalues(states):
    QValues = []
    
    for state in states:
        
        state = np.reshape(state, (3,3))
        QValue = np.zeros((3,3), dtype='float')
        QValue[state!=0]=-np.inf
        QValues.append(QValue.copy())
        
    return QValues

