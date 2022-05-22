# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:42:37 2022

@author: henrypapadatos
"""

import define_states
import numpy as np
import random

class QLearningPlayer():
    def __init__(self, exploration_level = 0.01, player='X'):
        self.states = define_states.define_states_graph()
        self.QValuesX = define_states.initialize_Qvalues(self.states)
        self.QValuesO = define_states.initialize_Qvalues(self.states)
        self.exploration_level = exploration_level
        self.reward = 0
        self.last_state = None
        self.last_action = None
        self.lr = 0.05
        self.discount_factor = 0.99
        
        
        if (player != 'X') and (player != 'O'):
            raise ValueError ("Wrong player type.")
        else:
            self.reward = 0
            self.player = player
        
    def set_player(self, player):
        if (player != 'X') and (player != 'O'):
            raise ValueError ("Wrong player type.")
        else:
            self.player = player
            
    def last_update(self,reward):
        self.reward = reward
        self.update_QValues(0)
            
    def update_QValues(self, next_max_QValues):
        
        #if this is the first action played, we don't update the QValues
        if self.last_state==None:
            return
        
        if self.player == 'X':
            QValues=self.QValuesX[self.last_state]
        else:
            QValues=self.QValuesO[self.last_state]
            
        QValue=QValues[self.last_action]
            
        delta_Qvalue = self.lr*(self.reward+self.discount_factor*next_max_QValues-QValue)
        
        if self.player == 'X':
            self.QValuesX[self.last_state][self.last_action]+=delta_Qvalue
        else:
            self.QValuesO[self.last_state][self.last_action]+=delta_Qvalue
        
        
    def act(self, board, train_mode=False):
        board = np.reshape(board.copy(), 9).tolist()
        state = self.states.index(board)
        
        if self.player == 'X':
            QValues=self.QValuesX[state] 
        else:
            QValues=self.QValuesO[state] 
            
        if train_mode:
            #find best QValues to update Qvalues for the last state
            max_QValue = np.amax(QValues)
            self.update_QValues(max_QValue)
        
        #in this case we don't explore
        if random.random()>self.exploration_level:
            action = np.where(QValues == np.amax(QValues))
            
        #in this case, we explore
        else:
            #find all possible actions
            action = np.where(QValues!=-np.inf)
            
        #take one possible action randomly among the one we wants
        action_idx = random.randint(0, len(action[0])-1)
        
        action = (action[0][action_idx], action[1][action_idx])
        
        if train_mode:
            self.last_state=state
            self.last_action=action
        
        return action
    
    

    
    