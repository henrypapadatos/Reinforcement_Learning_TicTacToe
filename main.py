# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:46:18 2022

@author: papad
"""
import numpy as np
import QLearning
from tic_env import TictactoeEnv, OptimalPlayer
from time import perf_counter

######PRINT NUMBER OF EQUALITIES DURING TRAINING ##########

def test_policy(Qplayer):
    env = TictactoeEnv()
    nb_test_play = 500
    nb_win = 0
    
    turns = ['X','O']
    
    for idx in range(len(turns)):
        Qplayer_sign = turns[idx]
        player_opt_sign = turns[not idx]
        
        Qplayer.set_player(Qplayer_sign)
        player_opt = OptimalPlayer(epsilon=0., player=player_opt_sign)
    
        for i in range(int(nb_test_play/2)):
            grid, _, __ = env.observe()
            for j in range(9):
                if env.current_player == player_opt.player:
                    move = player_opt.act(grid)
                else:
                    move = Qplayer.act(grid)
    
                grid, end, winner = env.step(move, print_grid=False)
                #print(winner)
    
                if end:
                    #print(grid)
                    #print("IN END", winner)
                    if winner==Qplayer_sign:
                        nb_win+=1
                    # print('-------------------------------------------')
                    # print('Game end, winner is player ' + str(winner))
                    # print('Optimal player = ' +  Turns[1])
                    # print('Q player = ' +  Turns[0])
                    # env.render()
                    env.reset()
                    break
        
    return (nb_win/nb_test_play)*100      

t1_start = perf_counter()


Qplayer = QLearning.QLearningPlayer()

Turns = np.array(['X','O'])

env = TictactoeEnv()

nb_eval = 40

nb_play = 10000

for k in range(nb_eval):
    print("EVAL: ", k)
    for i in range(nb_play):
        
        grid, _, __ = env.observe()
        
        #pick a player randomly
        Turns = Turns[np.random.permutation(2)]
        
        Qplayer.set_player(player=Turns[0])
        player_opt = OptimalPlayer(epsilon=0., player=Turns[1])
        
        for j in range(9):
            if env.current_player == player_opt.player:
                move = player_opt.act(grid)
            else:
                move = Qplayer.act(grid, train_mode=True)

            grid, end, winner = env.step(move, print_grid=False)

            if end:
                if winner==Qplayer.player:
                    reward = 1
                else:
                    reward = -1
                    
                Qplayer.last_update(reward)
                # print('-------------------------------------------')
                # print('Game end, winner is player ' + str(winner))
                # print('Optimal player = ' +  Turns[1])
                # print('Q player = ' +  Turns[0])
                # env.render()
                env.reset()
                break
        
        if not i%1000:
            print('epoch: '+str(i))

    print("Average reward: {:.01f}".format(test_policy(Qplayer))) 
    

   

t1_stop = perf_counter()
d_time = t1_stop-t1_start
print('Elapsed time : {:.02f}s'.format(d_time))  