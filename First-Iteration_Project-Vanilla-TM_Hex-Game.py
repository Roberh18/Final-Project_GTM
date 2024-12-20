import numpy as np
import math
from random import random, choice
import csv
import matplotlib.pyplot as plt
from hex_game import run_hex_game
from Heatmap import plot_memory_heatmaps
import os
import json



np.random.seed(7)

# Convert string values to boolean if relevant
def str_to_bool(str):
    if str.lower() in ('true', 'false'):
        return str.lower() == 'true'
    return str  # If str is not a version of T/F and is not a boolean, return as is. 


def visualize_board(state):
    board = [[state[f'cell{i}_{j}'] for j in range(7)] for i in range(7)]
    output = "\n  x x x x x x x\n"
    for i, row in enumerate(board):
        output += " " * i
        output += f"o"
        for cell in row:
            if cell == '1':
                output += " X"
            elif cell == '-1':
                output += " O"
            else:
                output += " ."
        output += "\n"
    return output


def print_game_states(dataset, winner):
    output = visualize_board(dataset[0])
    for i, state in enumerate(dataset[1:], 1):
        output += f"\nFinal State Game {i}, "
        output += f"winner: {'X' if winner[i] == '1' else 'O'}\n\n"
        output += visualize_board(state)
        output += "\n\n"
    return output


def remove_contradictions(condition):
    # Create a set to store unique cells
    unique_cells = set()
    non_contradictory = []
    for literal in condition:
        # Extract the cell part (e.g., 'cell1_1' from 'NOT cell1_1')
        cell = literal.replace('NOT ', '')
        # If we haven't seen this cell before, add it to unique_cells and non_contradictory
        if cell not in unique_cells:
            unique_cells.add(cell)
            non_contradictory.append(literal)
        else:
            opposite = f"NOT {cell}" if literal.startswith('NOT') else f"NOT {literal}"
            if opposite in non_contradictory:
                non_contradictory.remove(opposite)
                non_contradictory.append(literal)

    return non_contradictory


class Memory:
    
    def __init__(self, forget_value, memorize_value, memory, num_states):
        self.memory = memory
        self.forget_value = forget_value
        self.memorize_value = memorize_value
        self.num_states = num_states
    
    def get_memory(self):
        return self.memory
    
    def get_literals(self):
        return list(self.memory.keys())
    
    def get_condition(self):
        condition = []
        for literal in self.memory:
            if self.memory[literal] > (self.num_states * 0.4):
                condition.append(literal)
        return condition
        
    def memorize(self, literal):
        if random() <= self.memorize_value and self.memory[literal] < self.num_states:
            self.memory[literal] += 1
            
    def forget(self, literal):
        if random() <= self.forget_value and self.memory[literal] > 1:
            self.memory[literal] -= 1
            
    def memorize_always(self, literal):
        if  self.memory[literal] < self.num_states:
            self.memory[literal] += 1  


def evaluate_condition(observation, condition):
    result = all(
        (observation[literal] == '1' 
            if not literal.startswith('NOT ') 
            else observation[literal[4:]] != '1')
            for literal in condition)
    return result


def type_i_feedback(observation, memory):
    remaining_literals = memory.get_literals()
    if evaluate_condition(observation, memory.get_condition()):
        for feature in observation:
            # print("evaluate_condition(observation, memory.get_condition()) == True, observation=", observation, "memory.get_condition()=", memory.get_condition())
            # print()
            # print("observation[feature]: \n", observation[feature])
            # print()
            # print("remaining_literals: \n", remaining_literals)
            # print()
            if observation[feature] == '1':
                memory.memorize(feature)
                remaining_literals.remove(feature)
            elif observation[feature] == '-1':
                memory.memorize('NOT ' + feature)
                remaining_literals.remove('NOT ' + feature)
    for literal in remaining_literals:
        memory.forget(literal)

        
def type_ii_feedback(observation, memory):
    if evaluate_condition(observation, memory.get_condition()):
        for feature in observation:
            if observation[feature] == '-1':
                memory.memorize_always(feature)
            elif observation[feature] == '1':
                memory.memorize_always('NOT ' + feature)


def classify(observation, player1_rules, player2_rules):
    vote_sum = 0
    if evaluate_condition(observation, player1_rules.get_condition()):
        vote_sum += 1
    if evaluate_condition(observation, player2_rules.get_condition()):
        vote_sum -= 1
    return 1 if vote_sum > 0 else -1


def evaluate_model(X, y, player1_rules, player2_rules):
    correct = sum(1 for obs, true_winner in zip(X, y) if classify(obs, player1_rules, player2_rules) == true_winner)
    return correct / len(X)


def main():
    # The value of each cell is:
    #  1 if it was played by the first player ("X" or "red")
    # -1 if it was played by the second player ("O" or "blue")
    # 0 if the cell has not been played. 
    # There is also a winner column indicating the player who won the game. Since each row represents a complete game, and Hex cannot end in a draw, these values are always 1 or -1.
    
    # Winner prediction accuracy at
    #   a) End of game
    #   b) Two moves before the end
    #   c) Five moves before the end
    # • Initial submission: October 17
    # • Intermediate submission: November 7
    # • Final Submission: December 5
    
    observation_id = 0
    c = 0
    for i in range(3):
        if i == 0:
            c = "prediction model to predict winner at stage: end of game."
            observation_id = 0
        elif i == 1:
            c = "prediction model to predict winner at stage: two moves before the end."
            observation_id = 2
        elif i == 2:
            c = "prediction model to predict winner at stage: five moves before the end."
            observation_id = 5
        print("\nTraining an validating", c, "\n")
         
        forget_value = 0.1
        memorize_value = 1 - forget_value
        Nr_of_states = 1000
        Training_iterations = 20000
        player_1_wins = 0
        player_2_wins = 0
        Validation_iterations = 0
        Dataset_winner_list = []
        Accuracy_list = []
        Player_1_memory_list = []
        Player_2_memory_list = []
        Player_1_rules_list = []
        Player_2_rules_list = []
        Dataset_list = []
        ListOfKeys = ['cell0_0', 'cell0_1', 'cell0_2', 'cell0_3', 'cell0_4', 'cell0_5', 'cell0_6',
                    'cell1_0', 'cell1_1', 'cell1_2', 'cell1_3', 'cell1_4', 'cell1_5', 'cell1_6',
                    'cell2_0', 'cell2_1', 'cell2_2', 'cell2_3', 'cell2_4', 'cell2_5', 'cell2_6',
                    'cell3_0', 'cell3_1', 'cell3_2', 'cell3_3', 'cell3_4', 'cell3_5', 'cell3_6',
                    'cell4_0', 'cell4_1', 'cell4_2', 'cell4_3', 'cell4_4', 'cell4_5', 'cell4_6',
                    'cell5_0', 'cell5_1', 'cell5_2', 'cell5_3', 'cell5_4', 'cell5_5', 'cell5_6',
                    'cell6_0', 'cell6_1', 'cell6_2', 'cell6_3', 'cell6_4', 'cell6_5', 'cell6_6',
                    'winner']
        
        Dataset = []
        Predictions = []
        Accuracy = 0
        winner = ""
    
        while Accuracy <= 0.9:
            # Declare list of literals and initialize their starting position in memory
            literals_player_1 = {
                                'cell0_0':(int(Nr_of_states / 2)), 'NOT cell0_0':(int(Nr_of_states / 2)),
                                'cell0_1':(int(Nr_of_states / 2)), 'NOT cell0_1':(int(Nr_of_states / 2)),
                                'cell0_2':(int(Nr_of_states / 2)), 'NOT cell0_2':(int(Nr_of_states / 2)),
                                'cell0_3':(int(Nr_of_states / 2)), 'NOT cell0_3':(int(Nr_of_states / 2)),
                                'cell0_4':(int(Nr_of_states / 2)), 'NOT cell0_4':(int(Nr_of_states / 2)),
                                'cell0_5':(int(Nr_of_states / 2)), 'NOT cell0_5':(int(Nr_of_states / 2)),
                                'cell0_6':(int(Nr_of_states / 2)), 'NOT cell0_6':(int(Nr_of_states / 2)),
                                'cell1_0':(int(Nr_of_states / 2)), 'NOT cell1_0':(int(Nr_of_states / 2)),
                                'cell1_1':(int(Nr_of_states / 2)), 'NOT cell1_1':(int(Nr_of_states / 2)),
                                'cell1_2':(int(Nr_of_states / 2)), 'NOT cell1_2':(int(Nr_of_states / 2)),
                                'cell1_3':(int(Nr_of_states / 2)), 'NOT cell1_3':(int(Nr_of_states / 2)),
                                'cell1_4':(int(Nr_of_states / 2)), 'NOT cell1_4':(int(Nr_of_states / 2)),
                                'cell1_5':(int(Nr_of_states / 2)), 'NOT cell1_5':(int(Nr_of_states / 2)),
                                'cell1_6':(int(Nr_of_states / 2)), 'NOT cell1_6':(int(Nr_of_states / 2)),
                                'cell2_0':(int(Nr_of_states / 2)), 'NOT cell2_0':(int(Nr_of_states / 2)),
                                'cell2_1':(int(Nr_of_states / 2)), 'NOT cell2_1':(int(Nr_of_states / 2)),
                                'cell2_2':(int(Nr_of_states / 2)), 'NOT cell2_2':(int(Nr_of_states / 2)),
                                'cell2_3':(int(Nr_of_states / 2)), 'NOT cell2_3':(int(Nr_of_states / 2)),
                                'cell2_4':(int(Nr_of_states / 2)), 'NOT cell2_4':(int(Nr_of_states / 2)),
                                'cell2_5':(int(Nr_of_states / 2)), 'NOT cell2_5':(int(Nr_of_states / 2)),
                                'cell2_6':(int(Nr_of_states / 2)), 'NOT cell2_6':(int(Nr_of_states / 2)),
                                'cell3_0':(int(Nr_of_states / 2)), 'NOT cell3_0':(int(Nr_of_states / 2)),
                                'cell3_1':(int(Nr_of_states / 2)), 'NOT cell3_1':(int(Nr_of_states / 2)),
                                'cell3_2':(int(Nr_of_states / 2)), 'NOT cell3_2':(int(Nr_of_states / 2)),
                                'cell3_3':(int(Nr_of_states / 2)), 'NOT cell3_3':(int(Nr_of_states / 2)),
                                'cell3_4':(int(Nr_of_states / 2)), 'NOT cell3_4':(int(Nr_of_states / 2)),
                                'cell3_5':(int(Nr_of_states / 2)), 'NOT cell3_5':(int(Nr_of_states / 2)),
                                'cell3_6':(int(Nr_of_states / 2)), 'NOT cell3_6':(int(Nr_of_states / 2)),
                                'cell4_0':(int(Nr_of_states / 2)), 'NOT cell4_0':(int(Nr_of_states / 2)),
                                'cell4_1':(int(Nr_of_states / 2)), 'NOT cell4_1':(int(Nr_of_states / 2)),
                                'cell4_2':(int(Nr_of_states / 2)), 'NOT cell4_2':(int(Nr_of_states / 2)),
                                'cell4_3':(int(Nr_of_states / 2)), 'NOT cell4_3':(int(Nr_of_states / 2)),
                                'cell4_4':(int(Nr_of_states / 2)), 'NOT cell4_4':(int(Nr_of_states / 2)),
                                'cell4_5':(int(Nr_of_states / 2)), 'NOT cell4_5':(int(Nr_of_states / 2)),
                                'cell4_6':(int(Nr_of_states / 2)), 'NOT cell4_6':(int(Nr_of_states / 2)),
                                'cell5_0':(int(Nr_of_states / 2)), 'NOT cell5_0':(int(Nr_of_states / 2)),
                                'cell5_1':(int(Nr_of_states / 2)), 'NOT cell5_1':(int(Nr_of_states / 2)),
                                'cell5_2':(int(Nr_of_states / 2)), 'NOT cell5_2':(int(Nr_of_states / 2)),
                                'cell5_3':(int(Nr_of_states / 2)), 'NOT cell5_3':(int(Nr_of_states / 2)),
                                'cell5_4':(int(Nr_of_states / 2)), 'NOT cell5_4':(int(Nr_of_states / 2)),
                                'cell5_5':(int(Nr_of_states / 2)), 'NOT cell5_5':(int(Nr_of_states / 2)),
                                'cell5_6':(int(Nr_of_states / 2)), 'NOT cell5_6':(int(Nr_of_states / 2)),
                                'cell6_0':(int(Nr_of_states / 2)), 'NOT cell6_0':(int(Nr_of_states / 2)),
                                'cell6_1':(int(Nr_of_states / 2)), 'NOT cell6_1':(int(Nr_of_states / 2)),
                                'cell6_2':(int(Nr_of_states / 2)), 'NOT cell6_2':(int(Nr_of_states / 2)),
                                'cell6_3':(int(Nr_of_states / 2)), 'NOT cell6_3':(int(Nr_of_states / 2)),
                                'cell6_4':(int(Nr_of_states / 2)), 'NOT cell6_4':(int(Nr_of_states / 2)),
                                'cell6_5':(int(Nr_of_states / 2)), 'NOT cell6_5':(int(Nr_of_states / 2)),
                                'cell6_6':(int(Nr_of_states / 2)), 'NOT cell6_6':(int(Nr_of_states / 2))}
            literals_player_2 = literals_player_1.copy()
            
            # intialize memory for both classes
            Player_1_memory = Memory(forget_value, memorize_value, literals_player_1, Nr_of_states)
            Player_2_memory = Memory(forget_value, memorize_value, literals_player_2, Nr_of_states)
            
            for j in range(Training_iterations):
                    
                # Run the random hex game generator and get the data for the last five states of a random hex game.
                Dataset = run_hex_game()
                Dataset_list.append(Dataset[0])
                Dataset_winner_list.append(Dataset[0]['winner'])
                
                # Print visual representations of the board states
                # print_game_states(dataset)
                # Print Dataset
                # print(Dataset)
                
                if (Dataset[0]['winner'] == str(1)):
                    # print("Winner is player 1")
                    winner = "player_1"
                    player_1_wins += 1
                elif (Dataset[0]['winner'] == str(-1)):
                    # print("Winner is player 2")
                    winner = "player_2"
                    player_2_wins += 1
                
                OnePercent = (Training_iterations / 100)  
                if j % OnePercent == 0:
                    print("Training progress: ", ((j / OnePercent) + 1), "%, \t", end="")
                    print("Player 1 wins: ", round((player_1_wins / (j+1))*100, 2), "% of the games.")
                
                if winner == "player_1":
                    # Remove entry in a dictionary with key 'winner'
                    del Dataset[0]['winner']
                    coinflip = choice([0, 1])
                    if coinflip == 1:
                        type_i_feedback(Dataset[observation_id], Player_1_memory)
                        temp = Player_1_memory.get_memory()
                        Player_1_memory_list.append(temp)
                    elif coinflip == 0:
                        type_ii_feedback(Dataset[observation_id], Player_1_memory)
                        temp = Player_1_memory.get_memory()
                        Player_1_memory_list.append(temp)

                if winner == "player_2":
                    # Remove entry in a dictionary with key 'winner'
                    del Dataset[0]['winner']
                    coinflip = choice([0, 1])
                    if coinflip == 1:
                        type_i_feedback(Dataset[observation_id], Player_2_memory)
                        temp = Player_1_memory.get_memory()
                        Player_2_memory_list.append(temp)
                    elif coinflip == 0:
                        type_ii_feedback(Dataset[observation_id], Player_2_memory)
                        temp = Player_1_memory.get_memory()
                        Player_2_memory_list.append(temp)
                        
                # List for plotting memory
                Player_1_memory_list.append(Player_1_memory.get_memory())
                Player_2_memory_list.append(Player_2_memory.get_memory())
            
                
            print("Player_1_memory: \n", Player_1_memory.get_memory(), end="\n")
            print("Player_2_memory: \n", Player_2_memory.get_memory(), end="\n")
            
            # Filter out contradictary statements from conditions
            player_1_condition = remove_contradictions(Player_1_memory.get_condition())
            player_2_condition = remove_contradictions(Player_2_memory.get_condition())
            
            Player_1_rules_list.append(player_1_condition)
            Player_2_rules_list.append(player_2_condition)
            print("Player_1_memory.get_condition(): \n", player_1_condition, end="\n")
            print("Player_2_memory.get_condition(): \n", player_2_condition, end="\n")
            
            with open("AI_Fall_2024/IKT457-Learning-Systems/Assignment 3 - Project/Output.txt", 'a') as file:
                file.write("############################################# >>> NEW ENTRY <<< ##################################################\n")
                file.write(f"forget_value: {forget_value}\n")
                file.write(f"memorize_value: {memorize_value}\n")
                file.write(f"Nr_of_states: {Nr_of_states}\n")
                file.write(f"Training_iterations: {Training_iterations}\n")
                file.write(f"Validation_iterations: {Validation_iterations}\n")
                file.write(f"Accuracy: {Accuracy}\n")
                file.write(f"Player_1_memory.get_memory(): {Player_1_memory.get_memory()}\n")
                file.write(f"Player_2_memory.get_memory(): {Player_2_memory.get_memory()}\n")
                file.write(f"Player_1_memory.get_condition(): {player_1_condition}\n")
                file.write(f"Player_2_memory.get_condition(): {player_2_condition}\n")
                file.write(f"print_game_states(Dataset_list): {print_game_states(Dataset_list, Dataset_winner_list)}\n")
                file.write("\n\n\n")  
                file.write("##################################################################################################################")  
                file.write("\n\n\n")
            
            plot_memory_heatmaps(Player_1_memory.get_memory())
            plot_memory_heatmaps(Player_2_memory.get_memory())
                
            exit()   

    Accuracy = 0
    Player_1_rules = []
    Player_2_rules = []

    #while Accuracy < 0.90:
    
    #print(f"\nTraining model for {moves_before_end} moves before end...")
    
    while Accuracy < 0.8 and Iteration < 100:
        player1_rules, player2_rules = train_model(X_train, y_train, forget_value, memorize_value, num_iterations)
    
        Accuracy = evaluate_model(X_test, y_test, player1_rules, player2_rules)
        print(f"Iteration: {Iteration}, Accuracy {moves_before_end} moves before end: {Accuracy*100:.4f}%")
        Player_1_rules = player1_rules.get_condition()
        Player_2_rules = player2_rules.get_condition()
        Iteration += 1
    
    print("Player 1 rules:", Player_1_rules)
    print("Player 2 rules:", Player_2_rules)

            
if __name__ == "__main__":
    main()
