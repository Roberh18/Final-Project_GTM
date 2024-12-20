import os
import csv
import random
import time

# --------------------------
# Hyperparameters
# --------------------------
BOARD_DIM = 7              # Board dimension (e.g., 7 for a 7x7 Hex board)
NUM_GAMES = 20000          # Number of games to generate
RETURN_STATES_BEFORE_END = 0  # How many moves before the game ended should we return the state
MIN_OPEN_POSITIONS = round((((BOARD_DIM-2)*(BOARD_DIM-2))/2)+RETURN_STATES_BEFORE_END*2) + round(BOARD_DIM / 3)    # Minimum number of open positions required in the chosen state
USE_RANDOM_START_PLAYER = True  # If True, randomly choose the starting player each game. Otherwise, use player 0 always.

# --------------------------
# Hex Game Generation Code
# --------------------------

start_time = time.time()

# Directions for neighbor checks in hex (using axial neighbors)
neighbors = [-(BOARD_DIM + 2) + 1, -(BOARD_DIM + 2), -1, 1, (BOARD_DIM + 2), (BOARD_DIM + 2) - 1]

class HexGame:
    def __init__(self, board_dim=BOARD_DIM):
        self.board_dim = board_dim
        self.board = [0] * ((board_dim + 2) * (board_dim + 2) * 2)
        self.open_positions = []
        self.number_of_open_positions = 0
        self.moves = []
        self.connected = [0] * ((board_dim + 2) * (board_dim + 2) * 2)

    def init(self):
        # Initialize the board with border offsets
        self.open_positions = []
        for i in range(self.board_dim + 2):
            for j in range(self.board_dim + 2):
                idx = (i * (self.board_dim + 2) + j)
                self.board[idx * 2] = 0
                self.board[idx * 2 + 1] = 0

                # Positions inside the board are initially open
                if 0 < i < self.board_dim + 1 and 0 < j < self.board_dim + 1:
                    self.open_positions.append(idx)

                # Setup connected arrays for top and left borders
                if i == 0:
                    self.connected[idx * 2] = 1  # Top row connected for Red
                else:
                    self.connected[idx * 2] = 0

                if j == 0:
                    self.connected[idx * 2 + 1] = 1  # Left column connected for Blue
                else:
                    self.connected[idx * 2 + 1] = 0

        self.number_of_open_positions = self.board_dim * self.board_dim
        self.moves = []

    def connect(self, player, position):
        self.connected[position * 2 + player] = 1

        # Check if Red (player=0) reached bottom edge
        if player == 0 and position // (self.board_dim + 2) == self.board_dim:
            return 1

        # Check if Blue (player=1) reached right edge
        if player == 1 and position % (self.board_dim + 2) == self.board_dim:
            return 1

        for i in range(6):
            neighbor = position + neighbors[i]
            if self.board[neighbor * 2 + player] and not self.connected[neighbor * 2 + player]:
                if self.connect(player, neighbor):
                    return 1
        return 0

    def winner_check(self, player, position):
        # After placing a piece by player at position, check if it leads to a win
        for i in range(6):
            neighbor = position + neighbors[i]
            if self.connected[neighbor * 2 + player]:  # already connected position next to it
                return self.connect(player, position)
        return 0

    def place_piece_randomly(self, player):
        random_empty_position_index = random.randint(0, self.number_of_open_positions - 1)
        empty_position = self.open_positions[random_empty_position_index]
        self.board[empty_position * 2 + player] = 1
        # Move number is (total board cells - current number_of_open_positions) but we append to moves directly
        self.moves.append(empty_position)
        # Remove the chosen open position
        self.open_positions[random_empty_position_index] = self.open_positions[self.number_of_open_positions - 1]
        self.number_of_open_positions -= 1

        return empty_position

    def full_board(self):
        return self.number_of_open_positions == 0

    def get_state_dict(self):
        # Return the board state in the format cell{i}_{j}: 1/-1/0
        #  i,j from 0 to BOARD_DIM-1
        state_dict = {}
        for i in range(self.board_dim):
            for j in range(self.board_dim):
                index = ((i + 1) * (self.board_dim + 2) + (j + 1)) * 2
                # Red=board[index], Blue=board[index+1]
                if self.board[index] == 1:
                    state_dict[f'cell{i}_{j}'] = '1'
                elif self.board[index + 1] == 1:
                    state_dict[f'cell{i}_{j}'] = '-1'
                else:
                    state_dict[f'cell{i}_{j}'] = '0'
        return state_dict

    def get_open_positions_count(self):
        return self.number_of_open_positions

    def restore_state(self, move_count, start_player):
        # Save moves before re-init
        saved_moves = self.moves[:]
        self.init()  # Re-initialize the board
        self.moves = []  # Clear current moves and replay

        player = start_player
        # Replay moves up to move_count
        for m in range(move_count):
            if m >= len(saved_moves):
                # Safety check in case something is off
                break
            pos = saved_moves[m]
            self.board[pos * 2 + player] = 1
            self.moves.append(pos)

            # Update open_positions by removing pos
            if pos in self.open_positions:
                idx = self.open_positions.index(pos)
                self.open_positions[idx] = self.open_positions[self.number_of_open_positions - 1]
                self.number_of_open_positions -= 1
            else:
                # If for some reason pos is not in open_positions, just decrement count
                self.number_of_open_positions -= 1

            player = 1 - player

def run_single_hex_game(board_dim, min_open_positions, return_states_before_end, use_random_start):
    # Run a single hex game and return the chosen state
    # Initialize the game
    hg = HexGame(board_dim)
    hg.init()

    # Decide starting player based on hyperparameter
    if use_random_start:
        start_player = random.randint(0, 1)
    else:
        start_player = 0  # Always start with Red if not random
    player = start_player
    winner = -1
    move_count = 0

    # Play until winner or full board
    while not hg.full_board():
        position = hg.place_piece_randomly(player)
        move_count += 1
        if hg.winner_check(player, position):
            winner = player
            break
        player = 1 - player

    # If no winner and board is full - hex shouldn't tie, but if it does, skip
    if winner == -1 and hg.full_board():
        return None

    final_move_count = move_count  # total moves played
    chosen_move_count = final_move_count - return_states_before_end
    if chosen_move_count < 0:
        chosen_move_count = final_move_count

    # Restore the game to the chosen move count
    hg.restore_state(chosen_move_count, start_player)

    # Check if at the chosen state we have enough open positions
    if hg.get_open_positions_count() < min_open_positions:
        # Not enough open positions, skip this game
        return None

    # If still no winner, skip
    if winner == -1:
        return None

    # Determine winner label:
    # Red is player=0 -> winner='1'
    # Blue is player=1 -> winner='-1'
    state_dict = hg.get_state_dict()
    state_dict['winner'] = '1' if winner == 0 else '-1'
    return state_dict

if __name__ == "__main__":
    # Create output directory
    os.makedirs("generated_games", exist_ok=True)
    output_file = f"generated_games/hex_games_{NUM_GAMES}_size_{BOARD_DIM}_BeforeEnd-{RETURN_STATES_BEFORE_END}_OpenPos-{MIN_OPEN_POSITIONS}_Random-{USE_RANDOM_START_PLAYER}.csv"

    # Prepare CSV header
    headers = [f'cell{i}_{j}' for i in range(BOARD_DIM) for j in range(BOARD_DIM)] + ['winner']

    results = []
    # Keep generating until we have exactly NUM_GAMES states
    while len(results) < NUM_GAMES:
        game_state = run_single_hex_game(BOARD_DIM, MIN_OPEN_POSITIONS, RETURN_STATES_BEFORE_END, USE_RANDOM_START_PLAYER)
        if game_state is not None:
            results.append(game_state)
            if len(results) % 500 == 0:
                print(".", end="", flush=True)

    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    elapsed_time = time.time() - start_time
    print(f"\nSuccessfully generated {len(results)} states and saved to {output_file}.")
    print(f"These games took {elapsed_time:.2f} seconds to generate.")
