import numpy as np
import Deep_QLearning
from tic_env import TictactoeEnv, OptimalPlayer
from time import perf_counter
import random
import test_functions

random.seed(10)

t1_start = perf_counter()

nb_eval = 40
nb_play = 1000
exploration_level = 0.4

DQN_player = Deep_QLearning.DQN_Player()

Turns = np.array(['X','O'])

env = TictactoeEnv()

for k in range(nb_eval):
    print("EVAL: ", k)
    
    DQN_player.set_exploration_level(exploration_level)

    
    for i in range(nb_play):
        
        grid, _, __ = env.observe()
        
        #pick a player randomly
        #Turns = Turns[np.random.permutation(2)]
        #Turns = ['X','O']
        current_turns = Turns[np.array([k%2, (k+1)%2])]
        
        DQN_player.set_player(player=current_turns[0])
        player_opt = OptimalPlayer(epsilon=0.5, player=current_turns[1])

        for j in range(9):
            if env.current_player == player_opt.player:
                move = player_opt.act(grid)
            else:
                Valid_move_flag, move = DQN_player.act(grid, train_mode=True)
            
            #if the move played by our player is not available, 
            #the game is stopped and the reward=-1
            if Valid_move_flag:
                grid, end, winner = env.step(move, print_grid=False)
                
                
            else:
                reward = -1
                DQN_player.last_update(reward)
                env.reset()
                break
            
            if end:
                if winner==DQN_player.player:
                    reward = 1
                elif winner==player_opt.player:
                    reward = -1
                else: 
                    reward = 0
                DQN_player.last_update(reward)
                env.reset()
                break
        
        if not i%100:
            print('epoch: '+str(i))
    print("Average reward: {:.03f}".format(test_functions.test_DQN_policy(DQN_player))) 
    

   

t1_stop = perf_counter()
d_time = t1_stop-t1_start
print('Elapsed time : {:.02f}s'.format(d_time))  
