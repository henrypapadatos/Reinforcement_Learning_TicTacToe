import numpy as np
import random
import torch
from torch import nn
import copy

class DQN_Player():
    def __init__(self,exploration_level = 0.3, player='X'):
        self.buffer = []
        self.exploration_level= exploration_level
        self.last_state = None
        self.last_action = None
        self.reward = 0
        self.lr = 5e-4
        self.batch_size = 64
        self.update_target_rate = 500
        self.update_counter = 0
        self.discount_factor = 0.99
        self.max_buffer_size = 1e4
        #our implementation does not support biais!
        self.model = nn.Sequential(nn.Linear(18, 128, bias=False),
                                   nn.ReLU(),
                                   nn.Linear(128, 128, bias=False),
                                   nn.ReLU(),
                                   nn.Linear(128, 9, bias=False)).float()
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = nn.HuberLoss()
        
        if (player != 'X') and (player != 'O'):
            raise ValueError ("Wrong player type.")
        else:
            self.player = player
            
        return

    def set_player(self, player):
        if (player != 'X') and (player != 'O'):
            raise ValueError ("Wrong player type.")
        else:
            self.player = player
    
    def set_exploration_level(self, exploration_level):
        self.exploration_level = exploration_level
        
    def is_action_valid(self,action, state):
        i = int(action / 3)
        j = action % 3
        
        if state[0,i,j]==0. and state[1,i,j]==0.:
            return True
        else:
            return False

    def grid_to_state(self, grid):
        # nous sommes des 1 ou des -1?
        our_state = grid.copy()
        opponent_state = grid.copy()
        our_state[our_state!=1] = 0
        opponent_state[opponent_state!=-1] = 0
        opponent_state[opponent_state==-1] = 1
        return torch.tensor(np.array([our_state, opponent_state]), dtype= torch.float32)

    def addSequenceToBuffer(self,state):
        
        if self.last_state is None:
            return

        if len(self.buffer)>self.max_buffer_size:
            self.buffer.pop(0)
            
        sequence = {'state': self.last_state,
                    'action':self.last_action,
                    'reward':self.reward, 
                    'next_state': state}

        self.buffer.append(sequence)
        
    def define_target(self):
        self.target = copy.deepcopy(self.model)
    
    def create_batch(self):
        temp_batch = random.sample(self.buffer, k = self.batch_size)
        state = torch.empty((64,18))
        action = torch.empty((64,1))
        reward = torch.empty((64,1))
        next_state = torch.empty((64,18))
        
        for i in range(len(temp_batch)):
            state[i] = temp_batch[i]['state'].view(-1)
            action[i] = temp_batch[i]['action']
            reward[i] = temp_batch[i]['reward']
            next_state[i] = temp_batch[i]['next_state'].view(-1)

        return state, action, reward, next_state
        
    def train_model(self):
        
        #If the buffer has not 64 elements yet, we don't train
        if len(self.buffer)<self.batch_size:
            return
        
        state, action, reward, next_state = self.create_batch()
        
        #self.model.train()
        
        QValues = self.model(state)
        QValue = torch.gather(QValues, 1, action.long())
        
        with torch.no_grad():
            QValues_hat = self.target(state)
            QValue_hat,_ = QValues_hat.max(1)
            QValue_hat = QValue_hat.mul(self.discount_factor).view(-1,1)
        
        target = torch.add(reward, QValue_hat)
        
        loss = self.criterion(QValue, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def act(self, board, train_mode=False):
        
        #We want to always train the agent as if it is playing X(1)
        #Therefore, if it is currently playing O(-1), we invert all the 1 and -1
        #in the board so we come back to a situation where the agent is playing 'X'
        if self.player == 'O':
          board=board.copy()*-1

        state = self.grid_to_state(board)
        
        self.addSequenceToBuffer(state)

        QValues = self.model(state.view(-1))
        
        if train_mode:
            self.train_model()
            if self.update_counter == 500:
                self.define_target()
                self.update_counter = 0
        
        #in this case, we explore. the action is sampled randomly from available actions
        if train_mode and random.random()<self.exploration_level:
            action = np.where(board==0)
            action_idx = random.randint(0, len(action[0])-1)
            action = int(action[0][action_idx]*3 + action[1][action_idx])
   
        #in this case, we don't explore
        else:
            action = QValues.argmax().int().item()

        if train_mode:
            self.last_action = action
            self.last_state = state
            self.update_counter+=1
            
        if self.is_action_valid(action, state) == False:
            Valid_move_flag = False
            return Valid_move_flag, action
                    
        Valid_move_flag = True
        return Valid_move_flag, action
        
            
    def last_update(self,reward):
        self.reward = reward
        
        #since there is no biais, Qhat of Sj+1 = 0 if board is empty
        board = np.zeros((3,3))
        state = self.grid_to_state(board)
        self.addSequenceToBuffer(state)
        
        self.train_model()
        
        self.reward = 0
        self.last_action=None
        self.last_state=None