# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:42:37 2022

@author: henrypapadatos
"""
import numpy as np
import random

class QLearningPlayer():
    def __init__(self, exploration_level = 0.3, decreasing_exploration_rate= 1, 
                 decreasing_exploration_flag = False, player='X'):
        self.states={}
        self.exploration_level = exploration_level
        self.reward = 0
        self.last_state = None
        self.last_action = None
        self.lr = 0.05
        self.discount_factor = 0.99
        self.epsilon_min = 0.1
        self.epsilon_max = 0.8
        self.game_number = 0
        self.decreasing_exploration_rate = decreasing_exploration_rate
        self.decreasing_exploration_flag = decreasing_exploration_flag

        
        if (player != 'X') and (player != 'O'):
            raise ValueError ("Wrong player type.")
        else:
            self.player = player
        
    def set_player(self, player):
        if (player != 'X') and (player != 'O'):
            raise ValueError ("Wrong player type.")
        else:
            self.player = player
    
    def set_decreasing_exploration_rate(self, decreasing_exploration_rate):
        self.decreasing_exploration_rate = decreasing_exploration_rate
            
    def set_exploration_level(self, exploration_level):
        self.exploration_level = exploration_level
        
    def compute_exploration_level(self):
        """
        Compute the learning rate in the case of decreasing_exploration_flag=True
        lr = max(epsilon_min, epsilon_max*(1-game_number/decreasing_exploration_rate))
     
        Returns
        -------
        None.

        """
        self.exploration_level = max((self.epsilon_min, self.epsilon_max*(1-self.game_number/self.decreasing_exploration_rate)))
            
    def addBoardToStates(self, board, key):
        
        QValues = np.zeros((3,3), dtype='float')
        QValues[board!=0]=-np.inf
        
        self.states[key] = QValues
            
    def last_update(self,reward):
        self.reward = reward
        self.update_QValues(0)
        
        self.reward = 0
        self.last_action=None
        self.last_state=None

            
    def update_QValues(self, next_max_QValues):
        
        #if this is the first action played, we don't update the QValues
        if self.last_state==None:
            return
        
        #find Q
        QValue=self.states[self.last_state][self.last_action]
            
        #delta_Q = lr*(r+df*Q'-D)
        delta_Qvalue = self.lr*(self.reward+self.discount_factor*next_max_QValues-QValue)
        
        self.states[self.last_state][self.last_action] += delta_Qvalue
        
        
    def act(self, board, train_mode=False):
        
        #We want to always train the agent as if it is playing X(1)
        #Therefore, if it is currently playing O(-1), we invert all the 1 and -1
        #in the board so we come back to a situation where the agent is playing 'X'
        if self.player == 'O':
          board=board.copy()*-1;
        
        #transform to board into a key for the state dictionnary
        key = board.tobytes()
        
        #if the board is not currently in the dict, add it
        if not key in self.states: self.addBoardToStates(board, key)
        
        QValues = self.states[key]
            
        if train_mode:
            #find best QValues to update Qvalues for the last state
            max_QValue = np.amax(QValues)
            self.update_QValues(max_QValue)
            
        #Compute the exploration_level in the case of decreasing_exploration_flag=True
        if self.decreasing_exploration_flag:
            self.compute_exploration_level()
        
        #in this case we explore
        if train_mode and random.random()<self.exploration_level:
            #find all possible actions
            action = np.where(QValues!=-np.inf)
            
        #in this case, we don't explore
        else:
            action = np.where(QValues == np.amax(QValues))
            
        #take one possible action randomly among the one we want
        action_idx = random.randint(0, len(action[0])-1)
        
        action = (action[0][action_idx], action[1][action_idx])
        
        if train_mode:
            self.last_state=key
            self.last_action=action
            self.game_number+=1
        
        return action
    
    

    
    