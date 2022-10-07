# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:28:10 2020

@author: Me
"""
import numpy as np

def main():
    dim = 3
    dim2 = dim*dim
    #vet_board = np.zeros(dim2)
    
    empty = 0
    o = 1
    x = -1
    
    #Inicializa tabela Q
    Q = 0.4*np.ones((3**dim2,dim2))
    
    #Parametros
    N_episodes = 30000 #Robo irá jogar este nr de partidas
    alpha = 0.5        #taxa de aprendizagem
    gamma = 0.95       #
    max_epsilon = 1
    min_epsilon = 0.05
    decay_rate  = 0.001
    epsilon = 1
    
    for episodes in range(N_episodes):
        Q = play_one_episode(Q,o,x,epsilon,dim, dim2, empty,alpha,gamma)
        epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episodes)
        if epsilon%1000 == 0:     
            print(epsilon)
            
    while True:
        play_teste(Q, dim, dim2, empty, x, o)
        
"""Função testa robo"""
def play_teste(Q,dim,dim2,empty,x,o):
    vet_board = np.zeros(dim2)
    gameover = False
    p1 = o
    p2 = x 
    current_player = []
    state =  take_state(dim2,vet_board,empty,x,o)
    draw_board(vet_board,x,o,dim)
    while not gameover:
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        
        #Player atual faz uma jogada
        if current_player == p1: #robo aprendiz
            action = take_action(Q,state,0, dim2,vet_board,empty)
        else:#humano
           print("Faça sua jogada") 
           action = input()
           action = int(action)
        
        #Preenchimento do tabuleiro
        vet_board[action] = current_player
        
        #desenha o tabuleiro novamente
        draw_board(vet_board,x,o,dim)
        
        #A partida terminou?
        gameover, winner = game_over(vet_board,dim,x,o)
        
        if gameover==True:
            print(winner,'ganhou')
        
        #Passar para o novo estado
        new_state = take_state(dim2,vet_board,empty,x,o)
            
        #atualização estado:
        state = new_state
        
   


"""Joga uma partida do jogo da velha"""
def play_one_episode(Q,o,x,epsilon, dim, dim2,empty,alpha,gamma):
    vet_board = np.zeros(dim2)
    gameover = False
    p1 = o
    p2 = x 
    current_player = []
    recorded_s_a_r = []
    state =  take_state(dim2,vet_board,empty,x,o)
    
    while not gameover:
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        
        #Player atual faz uma jogada
        if current_player == p1: #robo aprendiz
            action = take_action(Q,state,epsilon, dim2,vet_board,empty)
        else:#robo com ações aleatórias
            action = take_action(Q,state,1, dim2,vet_board,empty)
        
        #Preenchimento do tabuleiro
        vet_board[action] = current_player
        
        #A partida terminou?
        gameover, winner = game_over(vet_board,dim,x,o)
        
        #recebe a recompensa
        reward = get_reward(gameover,winner, p1)
        
        #Passar para o novo estado
        new_state = take_state(dim2,vet_board,empty,x,o)
        
        #armazenamento  da sequencia de state-action-reward do aprendiz
        if current_player == p1:
            recorded_s_a_r.append((state,action,reward))
       
        #atualização estado:
        state = new_state
        
    maximum = 0
    for s_a_r in reversed(recorded_s_a_r):
        #Tabela Q
        s = s_a_r[0]
        a = s_a_r[1]
        r = s_a_r[2]
        Q[s,a] = (1-alpha)*Q[s,a] + alpha*(r + gamma*maximum)
        maximum = np.max(Q[s,:])
        
    return Q

"""Tomada de decisão"""
def take_action(Q,state,epsilon, dim2,vet_board,empty):
    r = np.random.rand()
    
    possible_actions = []
    for i in range(dim2):
        if vet_board[i]==empty:
            possible_actions.append(i)
    if r<=epsilon:
        n = len(possible_actions)
        index = np.random.choice(n)
        action = possible_actions[index]
        return action
    else: #uso das experiencias
        Q_vals = Q[state,:] 
        Q_possible = [Q_vals[i] for i in possible_actions]#valores de q das ações possiveis
        max_Q_possible = np.max(Q_possible)#maximo valor de q dentro das ações possiveis
        actions_max = [i for i in possible_actions if Q_vals[i] == max_Q_possible]   
        action = np.random.choice(actions_max)
        return action
        
"""representaçã dos estados possiveis do tabuleiro"""
def take_state(dim2,vet_board,empty,x,o):
    somatorio = 0
    for i in range(dim2):
        if vet_board[i]==empty:
            digit = 0
        elif vet_board[i]==o:
            digit = 1
        else:
            digit = 2
        somatorio = somatorio + digit*(3**i)
        
    state = somatorio
    return state


"""Recompensa"""

def get_reward(gameover,winner, p1):
    if gameover and winner==p1:
        reward = 1
        return reward
    elif gameover and winner == 'tie':
        reward = 0.5
        return reward
    else:
        reward = 0
        return reward
    
        
def game_over(vet_board, dim, x, o):
    mat_board = np.reshape(vet_board,(dim,dim))
    
    #verifica linhas
    for player in (x,o):
        for i in range(dim):
            if mat_board[i,:].sum() == player*dim: #verifica linhas
                winner = player
                return True, winner
            elif mat_board[:,i].sum() == player*dim: #verifica colunas
                winner = player
                return True, winner
                
    #verifica diagonais
    for player in (x,o):
        if np.sum(np.diag(mat_board)) == player*dim: #diagonal principal
            winner = player
            return True, winner
        elif np.sum(np.diag(np.fliplr(mat_board))) == player*dim: #diagonal oposta
            winner = player
            return True, winner
    
    #verifica se deu empate
    if np.all((mat_board==0) == False): #todos os campos não estão vazios?
        winner = None
        return True, winner
    
    #Jogo ainda não terminou
    winner = None
    return False, winner
            
def draw_board(vet_board,x,o,dim):
    mat_board = np.resize(vet_board,(dim,dim))
    
    print(" ", end="")
    print("  1", end="  ")
    print(" 2", end="  ")
    print(" 3", end="  ") 
    print("")
    print("-------------")
    
    for i in range(dim):
        print(str(i+1), end=" ")
        for j in range(dim):
            print(" ", end="")
            if mat_board[i,j] == x:
                print("X  ", end="")
            elif mat_board[i,j] == o:
                print("O  ", end="")
            else:
                print("-  ", end="")
        print("")    

    
    
main()
    
    
    
    
    