import numpy as np
import random
import torch
from torch import nn
import copy


class DQN_Player():
    def __init__(self,exploration_level = 0.3, player='X',decreasing_exploration_rate= 1, 
                 decreasing_exploration_flag = False, batch_size=64, buffer_size=1e4):
        self.buffer = []
        self.exploration_level= exploration_level
        self.decreasing_exploration_rate = decreasing_exploration_rate
        self.decreasing_exploration_flag = decreasing_exploration_flag
        self.epsilon_min = 0.1
        self.epsilon_max = 0.8
        self.game_number = 0
        self.last_state = None
        self.last_action = None
        self.reward = 0
        self.lr = 1e-4#5e-4
        self.batch_size = batch_size
        self.update_target_rate = 500
        self.update_counter = 0
        self.discount_factor = 0.99
        self.max_buffer_size = buffer_size
        self.model = nn.Sequential(nn.Linear(18, 128, bias=True),
                                   nn.ReLU(),
                                   nn.Linear(128, 128, bias=True),
                                   nn.ReLU(),
                                   nn.Linear(128, 9, bias=True))
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        #self.criterion = nn.HuberLoss()
        self.criterion = nn.SmoothL1Loss()
        
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
    
    def set_decreasing_exploration_rate(self, decreasing_exploration_rate):
        self.decreasing_exploration_rate = decreasing_exploration_rate
    
    def compute_exploration_level(self):
        """
        Compute the learning rate in the case of decreasing_exploration_flag=True
        lr = max(epsilon_min, epsilon_max*(1-game_number/decreasing_exploration_rate))
     
        Returns
        -------
        None.

        """
        self.exploration_level = max((self.epsilon_min, self.epsilon_max*(1-self.game_number/self.decreasing_exploration_rate)))
        
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

        if len(self.buffer)>=self.max_buffer_size:
            self.buffer.pop(0)
            
        sequence = {'state': self.last_state,
                    'action':self.last_action,
                    'reward':self.reward, 
                    'next_state': state}

        self.buffer.append(sequence)
        
    def define_target(self):
        #self.target = copy.deepcopy(self.model)
        self.target.load_state_dict(self.model.state_dict())
    
    def create_batch(self):
        temp_batch = random.sample(self.buffer, k = self.batch_size)
        state = torch.empty((self.batch_size,18))
        action = torch.empty((self.batch_size,1))
        reward = torch.empty((self.batch_size,1))
        next_state = torch.empty((self.batch_size,18))
        
        for i in range(len(temp_batch)):
            state[i] = temp_batch[i]['state'].view(-1)
            action[i] = temp_batch[i]['action']
            reward[i] = temp_batch[i]['reward']
            next_state[i] = temp_batch[i]['next_state'].view(-1)

        return state, action, reward, next_state
        
    def train_model(self):
        
        #If the buffer has not 64 elements yet, we don't train
        if len(self.buffer)<self.batch_size:
            loss = 0
            return loss
        
        state, action, reward, next_state = self.create_batch()
        
        self.model.train()
        
        QValues = self.model(state)
        QValue = torch.gather(QValues, 1, action.long())
        
        with torch.no_grad():
            QValues_hat = self.target(next_state)
            QValue_hat,_ = QValues_hat.max(1)
            QValue_hat = torch.nan_to_num(QValue_hat)
            QValue_hat = QValue_hat.mul(self.discount_factor).view(-1,1)    
            target = torch.add(reward, QValue_hat)
        
        loss = self.criterion(QValue, target)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1,1)
            
        self.optimizer.step()
        
        return loss.item()
        
    def act(self, board, train_mode=False):
        
        #We want to always train the agent as if it is playing X (1)
        #Therefore, if it is currently playing O (-1), we invert all the 1 and -1
        #in the board so we come back to a situation where the agent is playing 'X'
        if self.player == 'O':
          board=board.copy()*-1

        state = self.grid_to_state(board)
        
        with torch.no_grad():
            QValues = self.model(state.view(-1))
        
        if train_mode:
            self.addSequenceToBuffer(state)
            loss = self.train_model()
            if self.update_counter == 500:
                self.define_target()
                self.update_counter = 0

        #Compute the exploration_level in the case of decreasing_exploration_flag=True
        if self.decreasing_exploration_flag:
            self.compute_exploration_level()
        
        #in this case, we explore. the action is sampled randomly from available actions
        if train_mode and random.random()<self.exploration_level:
            action = np.where(board==0)
            action_idx = random.randint(0, len(action[0])-1)
            action = int(action[0][action_idx]*3 + action[1][action_idx])
   
        #in this case, we don't explore
        else:
            with torch.no_grad():
                action = QValues.argmax().int().item()

        if train_mode:
            self.last_action = action
            self.last_state = state
            self.update_counter+=1
            self.game_number+=1
        else:
            loss = 0
            
        Valid_move_flag = self.is_action_valid(action, state)
        
        return Valid_move_flag, action, loss
        
            
    def last_update(self,reward, train_mode=False):
        self.reward = reward
        
        #Compute the exploration_level in the case of decreasing_exploration_flag=True
        if self.decreasing_exploration_flag:
            self.compute_exploration_level()
        
        state = torch.empty(2,3,3) * float('nan')
        self.addSequenceToBuffer(state)
        
        self.train_model()
        
        self.reward = 0
        self.last_action=None
        self.last_state=None
        
        if train_mode:
            self.game_number += 1
        
########################################################################################################################"
########################################################################################################################"
########################################################################################################################"
########################################################################################################################"


class DQNSelf_Player():
    def __init__(self,exploration_level = 0.3 ,decreasing_exploration_rate= 1, decreasing_exploration_flag = False):
        self.buffer = []
        self.exploration_level= exploration_level
        self.decreasing_exploration_rate = decreasing_exploration_rate
        self.decreasing_exploration_flag = decreasing_exploration_flag
        self.epsilon_min = 0.1
        self.epsilon_max = 0.8
        self.game_number = 0
        self.last_state_x = None
        self.last_action_x = None
        self.last_state_o = None
        self.last_action_o= None
        self.reward = 0
        self.lr = 1e-4#5e-4
        self.batch_size = 64
        self.update_target_rate = 500
        self.update_counter = 0
        self.discount_factor = 0.99
        self.max_buffer_size = 1e4
        #our implementation does not support biais!
        self.model = nn.Sequential(nn.Linear(18, 128, bias=True),
                                   nn.ReLU(),
                                   nn.Linear(128, 128, bias=True),
                                   nn.ReLU(),
                                   nn.Linear(128, 9, bias=True))
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        #self.criterion = nn.HuberLoss()
        self.criterion = nn.SmoothL1Loss()
        
        
    def set_exploration_level(self, exploration_level):
        self.exploration_level = exploration_level
    
    def set_decreasing_exploration_rate(self, decreasing_exploration_rate):
        self.decreasing_exploration_rate = decreasing_exploration_rate
    
    def compute_exploration_level(self):
        """
        Compute the learning rate in the case of decreasing_exploration_flag=True
        lr = max(epsilon_min, epsilon_max*(1-game_number/decreasing_exploration_rate))
     
        Returns
        -------
        None.

        """
        self.exploration_level = max((self.epsilon_min, self.epsilon_max*(1-self.game_number/self.decreasing_exploration_rate)))
        
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

    def addSequenceToBuffer(self,state, player):
        
        #if this is the first action played, we don't update the QValues
        if player == 'O' and self.last_action_o is not None:
            last_state = self.last_state_o
            last_action = self.last_action_o
        elif player == 'X' and self.last_action_x is not None:
            last_state = self.last_state_x
            last_action = self.last_action_x
        else:
            return

        if len(self.buffer)>=self.max_buffer_size:
            self.buffer.pop(0)
            
        sequence = {'state': last_state,
                    'action':last_action,
                    'reward':self.reward, 
                    'next_state': state}

        self.buffer.append(sequence)
        
    def define_target(self):
        #self.target = copy.deepcopy(self.model)
        self.target.load_state_dict(self.model.state_dict())
    
    def create_batch(self):
        temp_batch = random.sample(self.buffer, k = self.batch_size)
        state = torch.empty((self.batch_size,18))
        action = torch.empty((self.batch_size,1))
        reward = torch.empty((self.batch_size,1))
        next_state = torch.empty((self.batch_size,18))
        
        for i in range(len(temp_batch)):
            state[i] = temp_batch[i]['state'].view(-1)
            action[i] = temp_batch[i]['action']
            reward[i] = temp_batch[i]['reward']
            next_state[i] = temp_batch[i]['next_state'].view(-1)

        return state, action, reward, next_state
        
    def train_model(self):
        
        #If the buffer has not 64 elements yet, we don't train
        if len(self.buffer)<self.batch_size:
            loss = 0
            return loss
        
        state, action, reward, next_state = self.create_batch()
        
        self.model.train()
        
        QValues = self.model(state)
        QValue = torch.gather(QValues, 1, action.long())
        
        with torch.no_grad():
            QValues_hat = self.target(next_state)
            QValue_hat,_ = QValues_hat.max(1)
            QValue_hat = torch.nan_to_num(QValue_hat)
            QValue_hat = QValue_hat.mul(self.discount_factor).view(-1,1)    
            target = torch.add(reward, QValue_hat)
        
        loss = self.criterion(QValue, target)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1,1)
            
        self.optimizer.step()
        
        return loss.item()
        
    def act(self, board, player, train_mode=False):
        
        #We want to always train the agent as if it is playing X (1)
        #Therefore, if it is currently playing O (-1), we invert all the 1 and -1
        #in the board so we come back to a situation where the agent is playing 'X'
        if player == 'O':
          board=board.copy()*-1

        state = self.grid_to_state(board)
        
        with torch.no_grad():
            QValues = self.model(state.view(-1))
        
        if train_mode:
            self.addSequenceToBuffer(state, player)
            loss = self.train_model()
            if self.update_counter == 500:
                self.define_target()
                self.update_counter = 0

        #Compute the exploration_level in the case of decreasing_exploration_flag=True
        if self.decreasing_exploration_flag:
            self.compute_exploration_level()
        
        #in this case, we explore. the action is sampled randomly from available actions
        if train_mode and random.random()<self.exploration_level:
            action = np.where(board==0)
            action_idx = random.randint(0, len(action[0])-1)
            action = int(action[0][action_idx]*3 + action[1][action_idx])
   
        #in this case, we don't explore
        else:
            with torch.no_grad():
                action = QValues.argmax().int().item()

        if train_mode:
            if player == 'O':
                self.last_state_o=state
                self.last_action_o=action
            else:
                self.last_state_x=state
                self.last_action_x=action
            self.update_counter+=1
            self.game_number+=1
        else:
            loss = 0
            
        Valid_move_flag = self.is_action_valid(action, state)
        
        return Valid_move_flag, action, loss
        
            
    def last_update(self,reward,player,train_mode = False):
        self.reward = reward
        
        #Compute the exploration_level in the case of decreasing_exploration_flag=True
        if self.decreasing_exploration_flag:
            self.compute_exploration_level()
        
        state = torch.empty(2,3,3) * float('nan')
        self.addSequenceToBuffer(state, player)
        
        self.train_model()
        
        self.reward = 0        
        if player == 'O':
            self.last_action_o=None
            self.last_state_o=None
        else:
            self.last_action_x=None
            self.last_state_x=None
        
        if train_mode:
            self.game_number+=1
        