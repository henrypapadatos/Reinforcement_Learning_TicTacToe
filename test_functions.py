# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:46:18 2022

@author: papad
"""
import numpy as np
import QLearning
from tic_env import TictactoeEnv, OptimalPlayer
from time import perf_counter
import random

random.seed(10)

######PRINT NUMBER OF EQUALITIES DURING TRAINING ##########

def test_policy(Qplayer, expert_exploration_level=0):
    """
    Compute a metric to asses the performance of our model. 

    Parameters
    ----------
    Qplayer : QLearningPlayer
        Agent that we want to test.
    expert_exploration_level : int, optional
        exploration rate that the expert will use. If=0 the expert plays the optimal policy.
        If=1, the expert plays random moves . The default is 0.

    Returns
    -------
    float
        (#win - #loss)/#game player. If Qplayer is good, this value should be close to 1 
         if expert_exploration_level = 0 and it should be close to 0 if expert_exploration_level = 1

    """
    env = TictactoeEnv()
    nb_test_play = 500
    nb_win = 0
    nb_loss = 0
    
    turns = ['X','O']
      
    for idx in range(len(turns)):
        Qplayer_sign = turns[idx]
        player_opt_sign = turns[not idx]
        
        Qplayer.set_player(Qplayer_sign)
        player_opt = OptimalPlayer(epsilon=expert_exploration_level, player=player_opt_sign)
    
        for i in range(int(nb_test_play/2)):
            grid, _, __ = env.observe()
            for j in range(9):
                if env.current_player == player_opt.player:
                    move = player_opt.act(grid)
                else:
                    move = Qplayer.act(grid)
    
                grid, end, winner = env.step(move, print_grid=False)
                                
                if end:
                    if winner==Qplayer.player:
                        nb_win+=1
                    elif winner==player_opt.player:
                        nb_loss+=1
                    env.reset()
                    break
                
    return (nb_win-nb_loss)/nb_test_play    

if __name__ == "__main__": 
    t1_start = perf_counter()
    
    nb_eval = 40
    nb_play = 250
    exploration_level = 0.4
    
    Qplayer = QLearning.QLearningPlayer(decreasing_exploration_rate=1, decreasing_exploration_flag=True)
    
    Turns = np.array(['X','O'])
    
    env = TictactoeEnv()
    
    
    for k in range(nb_eval):
        print("EVAL: ", k)
        
        for i in range(nb_play):
            
            grid, _, __ = env.observe()
            
            #pick a player randomly
            Turns = Turns[np.random.permutation(2)]
            #Turns = ['X','O']
            
            Qplayer.set_player(player=Turns[0])
            player_opt = OptimalPlayer(epsilon=0.5, player=Turns[1])
            
            for j in range(9):
                if env.current_player == player_opt.player:
                    move = player_opt.act(grid)
                else:
                    move = Qplayer.act(grid, train_mode=True)
    
                grid, end, winner = env.step(move, print_grid=False)
    
                # env.render()
    
                if end:
                    if winner==Qplayer.player:
                        reward = 1
                    elif winner==player_opt.player:
                        reward = -1
                    else: 
                        reward = 0
                        
                    Qplayer.last_update(reward)
                    # print('-------------------------------------------')
                    # print('Game end, winner is player ' + str(winner))
                    # env.render()
                    env.reset()
                    break
            
            if not i%1000:
                print('epoch: '+str(i))
    
        print("Average reward: {:.03f}".format(test_policy(Qplayer, expert_exploration_level=0))) 
        
    
       
    
    t1_stop = perf_counter()
    d_time = t1_stop-t1_start
    print('Elapsed time : {:.02f}s'.format(d_time))  