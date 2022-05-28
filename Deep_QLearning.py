import define_states
import numpy as np
import random
import torch

class DQN_Player():
    def __init__(self,exploration_level = 0.3, player='X'):
        self.buffer = []
        return

    def set_player(self, player):
        if (player != 'X') and (player != 'O'):
            raise ValueError ("Wrong player type.")
        else:
            self.player = player
    
    def set_exploration_level(self, exploration_level):
        self.exploration_level = exploration_level

    def grid_to_state(self, grid):
        # nous sommes des 1 ou des -1?
        our_state = grid.copy()
        opponent_state = grid.copy()
        our_state[our_state!=1] = 0
        opponent_state[opponent_state!=-1] = 0
        return torch.tensor([our_state, opponent_state])

    def addStateToBuffer(self,sequence):

        if len(self.buffer)>1e4:
            self.buffer.pop(0)

        self.buffer.append(sequence)

    def act(self, board, train_mode=False):
        
        #We want to always train the agent as if it is playing X(1)
        #Therefore, if it is currently playing O(-1), we invert all the 1 and -1
        #in the board so we come back to a situation where the agent is playing 'X'
        if self.player == 'O':
          board=board.copy()*-1

        states = self.grid_to_state(board)

        # action = model.forward(states)

        if train_mode:
            # First stored sequence at begining of training
            if self.last_state:
                ###sequence = sequence = {'state': states, 'next_state': next_states, 'reward':reward, 'action':action}
                self.addStateToBuffer(sequence)
            #find best QValues to update Qvalues for the last state
                
            #transform to board into a key for the state dictionnary

            ### train to obtain Q####

            if random.random()<self.exploration_level:
            #find all possible actions
            action = np.where(QValues!=-np.inf)

            """

            EXEMPLE 

            states = torch.zeros(2,3,3)
            next_states = torch.rand(2,3,3)
            reward = 1
            action = (1,2)
            """
            ###sequence = sequence = {'state': states, 'next_state': next_states, 'reward':reward, 'action':action}
            
            #Add sequence to buffer 
            
        
            
        #in this case, we don't explore
        else:
            action = np.where(QValues == np.amax(QValues))
            
        #take one possible action randomly among the one we want
        action_idx = random.randint(0, len(action[0])-1)
        
        action = (action[0][action_idx], action[1][action_idx])
        
        if train_mode:
            self.last_state=key
            self.last_action=action
        
        return action
    
            
    def last_update(self,reward):
        self.reward = reward
        self.update_QValues(0)
        
        self.reward = 0
        self.last_action=None
        self.last_state=None