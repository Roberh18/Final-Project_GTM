import numpy as np
import pandas as pd
import os
import sys
import shutil
import logging
import datetime
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import time
import heapq
import matplotlib.patches as patches



TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

# Extract the filename without extension
script_path = os.path.abspath(__file__)
script_name = os.path.splitext(os.path.basename(script_path))[0]
folder_name = script_name.rsplit("_v", 1)[0] 
os.makedirs(folder_name, exist_ok=True)
for item in os.listdir(folder_name):
    item_path = os.path.join(folder_name, item)
    try:
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path) 
    except Exception as e:
        print(f"Failed to delete {item_path}. Reason: {e}")

'''
Dataset Paths:
Generator/generated_games/hex_games_20000_size_5_BeforeEnd-0_OpenPos-9_Random-True.csv
Generator/generated_games/hex_games_20000_size_5_BeforeEnd-3_OpenPos-14_Random-True.csv
Generator/generated_games/hex_games_20000_size_5_BeforeEnd-5_OpenPos-18_Random-True.csv
Generator/generated_games/hex_games_20000_size_7_BeforeEnd-0_OpenPos-14_Random-True.csv
Generator/generated_games/hex_games_20000_size_7_BeforeEnd-3_OpenPos-17_Random-True.csv
Generator/generated_games/hex_games_20000_size_7_BeforeEnd-5_OpenPos-19_Random-True.csv
Generator/generated_games/hex_games_20000_size_9_BeforeEnd-0_OpenPos-26_Random-True.csv
Generator/generated_games/hex_games_20000_size_9_BeforeEnd-3_OpenPos-29_Random-True.csv
Generator/generated_games/hex_games_20000_size_9_BeforeEnd-5_OpenPos-31_Random-True.csv
Generator/generated_games/hex_games_20000_size_11_BeforeEnd-0_OpenPos-40_Random-True.csv
Generator/generated_games/hex_games_20000_size_11_BeforeEnd-3_OpenPos-46_Random-True.csv
Generator/generated_games/hex_games_20000_size_11_BeforeEnd-5_OpenPos-50_Random-True.csv
'''


# ----------------------------
# Hyperparameters and Configuration
# ----------------------------

# General Configuration
VERSION = "1.2"
BOARD_SIZE = 7  # Size of the Hex board (e.g., 7 for 7x7)
DATA_FILE_PATH = 'Generator/generated_games/hex_games_20000_size_7_BeforeEnd-0_OpenPos-14_Random-True.csv'

# Training Parameters
EPOCHS = 20
NUMBER_OF_CLAUSES = 100 * (BOARD_SIZE**2)
T = round(NUMBER_OF_CLAUSES / 1.65)
s = 1.414
S_INITIAL = s
DEPTH = 3
DOUBLE_HASHING = False

# Data Parameters
DATA_REDUCTION_FACTOR = 0.5
VAL_TEST_SPLIT_RATIO = 0.3
RANDOM_STATE = 42

# Additional Edge Features
ADDITIONAL_EDGES = False

# Hyperparameter Tuning
ENABLE_TUNING_S = False
MAX_TRIALS = 25
EPOCHS_PER_TRIAL = 30
INCREMENT_S_PER_TRIAL_BY = 0.5

# Feature Selection Flags
ENABLE_CHEAT = False
ENABLE_MID_CTRL = True
ENABLE_EDGE_PIECE_COUNT = True
ENABLE_PIECE_COUNT = True
ENABLE_NEIGHBOR_COUNT = True
ENABLE_CLUSTER_STRENGTH = True
ENABLE_DIRECTIONAL_DOMINANCE = True
ENABLE_CLUSTER_ELONGATION = True
ENABLE_LONGEST_CHAIN = True
ENABLE_CRITICAL_PIECES = True
ENABLE_CONNECTIVITY_SCORE = True
ENABLE_BALANCE_OF_CONTROL = True


FEATURE_SELECTION = [
    ENABLE_CHEAT,
    ENABLE_MID_CTRL,
    ENABLE_EDGE_PIECE_COUNT,
    ENABLE_PIECE_COUNT,
    ENABLE_NEIGHBOR_COUNT,
    ENABLE_CLUSTER_STRENGTH,
    ENABLE_WEIGHTED_PATH,
    ENABLE_DIRECTIONAL_DOMINANCE,
    ENABLE_CLUSTER_ELONGATION,
    ENABLE_LONGEST_CHAIN,
    ENABLE_CRITICAL_PIECES,
    ENABLE_CONNECTIVITY_SCORE,
    ENABLE_BALANCE_OF_CONTROL
]


'''
MAPPING NUMBERS TO LITERALS
['X', 'O', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z', 'Connected', 'Not Connected']
  1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27       28             29
  
total_features = number_of_nodes * number_of_features_per_node
# For a 5x5 board with 27 symbols:
total_features = 25 * 29 = 725

'''

# ----------------------------
# Symbol Definitions
# ----------------------------

# Define feature-symbol mappings
FEATURE_SYMBOLS = {
    0: ['0', '1'],  # ENABLE_CHEAT
    1: ['A', 'B'],  # ENABLE_MID_CTRL
    2: ['C', 'D'],  # ENABLE_EDGE_PIECE_COUNT
    3: ['E', 'F'],  # ENABLE_PIECE_COUNT
    4: ['G', 'H'],  # ENABLE_NEIGHBOR_COUNT
    5: ['I', 'J'],  # ENABLE_CLUSTER_STRENGTH
    6: ['K', 'L'],  # ENABLE_DIRECTIONAL_DOMINANCE
    7: ['M', 'N'],  # ENABLE_CLUSTER_ELONGATION
    8: ['P', 'Q'],  # ENABLE_LONGEST_CHAIN
    9: ['R', 'S'],  # ENABLE_CRITICAL_PIECES
    10: ['T', 'U'],  # ENABLE_CONNECTIVITY_SCORE
    11: ['V', 'W'],  # ENABLE_BALANCE_OF_CONTROL
}

# Initialize symbols with basic cell states
symbols = ['X', 'O', ' ']

# Add additional symbols based on enabled features
logging_info_symbols = "\n\nBuilding list of symbols..."
print(logging_info_symbols)
logging_info_symbols = logging_info_symbols.strip()
logging.getLogger().setLevel(logging.INFO)

for index, is_enabled in enumerate(FEATURE_SELECTION):
    if is_enabled:
        symbols.extend(FEATURE_SYMBOLS[index])

if ADDITIONAL_EDGES:
    symbols.extend(["BuddyX", "BuddyO", "Empty Buddy", "Not Buddy"])
elif not ADDITIONAL_EDGES:
    symbols.extend(["Connected", "Not Connected"])
    

print("\n")
print(f"Symbols used in encoding: {symbols}\n")

HYPERVECTOR_SIZE = BOARD_SIZE * BOARD_SIZE * len(symbols)
HYPERVECTOR_BITS = 2
MESSAGE_SIZE = BOARD_SIZE * BOARD_SIZE * len(symbols)
MESSAGE_BITS = 2
MAX_INCLUDED_LITERALS = BOARD_SIZE * BOARD_SIZE

# ----------------------------
# Logging Setup
# ----------------------------

def setup_logging(epochs, version):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join("logs", f"training_log_epochs-{epochs}_SW-{version}_{timestamp}.log")
    
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    return timestamp

# Initialize logging
timestamp = setup_logging(EPOCHS, VERSION)

# Log hyperparameters and setup info
logging.info("Starting training run...")
logging.info(f"VERSION: {VERSION}")
logging.info("Hyperparameters and configuration:")
logging.info(f"epochs = {EPOCHS}")
logging.info(f"number_of_clauses = {NUMBER_OF_CLAUSES}")
logging.info(f"s_initial = {S_INITIAL}")
logging.info(f"T = {T}")
logging.info(f"depth = {DEPTH}")
logging.info(f"hypervector_size = {HYPERVECTOR_SIZE}")
logging.info(f"hypervector_bits = {HYPERVECTOR_BITS}")
logging.info(f"message_size = {MESSAGE_SIZE}")
logging.info(f"message_bits = {MESSAGE_BITS}")
logging.info(f"double_hashing = {DOUBLE_HASHING}")
logging.info(f"max_included_literals = {MAX_INCLUDED_LITERALS}")
logging.info(f"board_size = {BOARD_SIZE}")
logging.info(f"data_reduction_factor = {DATA_REDUCTION_FACTOR}")
logging.info(f"val_test_split_ratio = {VAL_TEST_SPLIT_RATIO}")
logging.info(f"random_state = {RANDOM_STATE}")
logging.info(f"FEATURE_SELECTION = {FEATURE_SELECTION}")
logging.info(f"Symbols = {symbols}")
print("\n")

# ----------------------------
# 2. Data Preparation
# ----------------------------

def load_and_prepare_data(data_file_path, board_size, data_reduction_factor, random_state):
    try:
        data = pd.read_csv(data_file_path)
        print(f"Data loaded successfully from {data_file_path}.")
    except FileNotFoundError:
        print(f"Error: The file {data_file_path} was not found.")
        sys.exit(1)
    
    # Reduce dataset size
    total_samples = len(data)
    reduced_samples = int(total_samples * data_reduction_factor)
    data = data.sample(n=reduced_samples, random_state=random_state).reset_index(drop=True)
    print(f"Dataset reduced to {reduced_samples} samples.")
    
    # Shuffle the dataset
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("Dataset shuffled successfully.")
    
    # Verify the dataset structure
    expected_columns = [f'cell{row}_{col}' for row in range(board_size) for col in range(board_size)] + ['winner']
    if list(data.columns) != expected_columns:
        print("Error: The dataset columns do not match the expected structure.")
        print(f"Expected columns: {expected_columns}")
        print(f"Found columns: {list(data.columns)}")
        sys.exit(1)
    else:
        print("Dataset columns verified.")
    
    # Extract feature columns
    feature_columns = [f'cell{row}_{col}' for row in range(board_size) for col in range(board_size)]
    X = data[feature_columns].values
    y = data['winner'].values
    
    logging.info(f"Total samples: {X.shape[0]}")
    logging.info(f"Feature shape: {X.shape[1]}")  # Should be board_size^2
    
    # Encode features
    X_cat = encode_features(X, board_size)
    
    # Encode labels
    y_encoded = encode_labels(y)
    
    return X_cat, y_encoded

def encode_features(X_numeric, board_size):
    '''
    Converts numerical cell values to categorical symbols.
    
    Mapping:
        1  -> 'X' (Red)
        -1 -> 'O' (Blue)
        0  -> ' ' (Empty)
    
    Parameters:
        X_numeric (np.ndarray): Numerical feature matrix.
        board_size (int): Size of the Hex board.
    
    Returns:
        np.ndarray: Categorical feature matrix.
    '''
    symbol_mapping = {1: 'X', -1: 'O', 0: ' '}
    X_categorical = np.vectorize(symbol_mapping.get)(X_numeric).astype(str)
    unique_symbols = np.unique(X_categorical)
    print(f"Unique symbols after encoding: {unique_symbols}")
    logging.info(f"Unique symbols in raw data: {unique_symbols}")
    return X_categorical

def encode_labels(y):
    '''
    Encodes winner labels from 1 and -1 to 0 and 1.
    
    Mapping:
        1  -> 0 (Red Wins)
        -1 -> 1 (Blue Wins)
    
    Parameters:
        y (np.ndarray): Original labels.
    
    Returns:
        np.ndarray: Encoded labels.
    '''
    label_mapping = {1: 0, -1: 1}  # Red:0, Blue:1
    y_encoded = np.vectorize(label_mapping.get)(y)
    return y_encoded

# ----------------------------
# 3. Feature Extraction
# ----------------------------

def get_neighbors(row, col, board_size=7):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < board_size and 0 <= c < board_size:
            yield r, c

def calculate_mid_control_categorical(board, board_size=7):
    mid_start = board_size // 2 - 1
    mid_end = board_size // 2 + 2  
    
    board_2d = board.reshape((board_size, board_size))
    mid_matrix = board_2d[mid_start:mid_end, mid_start:mid_end]
    
    red_count = np.sum(mid_matrix == 'X')
    blue_count = np.sum(mid_matrix == 'O')
    
    if red_count > blue_count:
        return 'A'  # Red controls
    else:
        return 'B'  # Blue controls

def calculate_edge_piece_count(board, board_size=7):
    edge_indices = set()
    
    # Top and bottom edges
    edge_indices.update(range(board_size))  # Top
    edge_indices.update(range(board_size * (board_size - 1), board_size**2))  # Bottom
    
    # Left and right edges
    for row in range(board_size):
        edge_indices.add(row * board_size)  # Left
        edge_indices.add(row * board_size + (board_size - 1))  # Right

    red_count = sum(1 for idx in edge_indices if board[idx] == 'X')
    blue_count = sum(1 for idx in edge_indices if board[idx] == 'O')
    if red_count > blue_count:
        return 'D'  # Red controls
    else:
        return 'C'  # Blue controls

def calculate_piece_count(board):
    red_count = np.sum(board == 'X')
    blue_count = np.sum(board == 'O')
    if red_count > blue_count:
        return 'E'  # Red controls
    else:
        return 'F'  # Blue controls

def calculate_high_neighbor_count(board, board_size=7):
    board_2d = board.reshape((board_size, board_size))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    
    def count_neighbors(row, col, player):
        count = 0
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < board_size and 0 <= c < board_size and board_2d[r, c] == player:
                count += 1
        return count

    red_count = 0
    blue_count = 0
    for row in range(board_size):
        for col in range(board_size):
            if board_2d[row, col] == 'X' and count_neighbors(row, col, 'X') >= 2:
                red_count += 1
            elif board_2d[row, col] == 'O' and count_neighbors(row, col, 'O') >= 2:
                blue_count += 1

    if red_count > blue_count:
        return 'G'  # Red controls
    else:
        return 'H'  # Blue controls

def compute_weighted_shortest_path(board_2d, player, board_size):
    opponent = 'O' if player == 'X' else 'X'
    cost_grid = np.full((board_size, board_size), 1, dtype=int)  
    cost_grid[board_2d == player] = 0 
    cost_grid[board_2d == opponent] = 999  # Opponent's cells have a large cost

    # Determine starting and ending edges based on the player
    if player == 'X':  # Top-to-bottom path
        start_points = [(0, col) for col in range(board_size)]
        end_points = [(board_size - 1, col) for col in range(board_size)]
    else:  # Left-to-right path
        start_points = [(row, 0) for row in range(board_size)]
        end_points = [(row, board_size - 1) for row in range(board_size)]

    # Initialize priority queue for Dijkstra's algorithm
    pq = []
    for start in start_points:
        heapq.heappush(pq, (0, start))  # (cost, position)

    visited = set()

    # Dijkstra's algorithm to find the shortest path
    while pq:
        current_cost, (x, y) = heapq.heappop(pq)
        if (x, y) in visited:
            continue
        visited.add((x, y))

        # If we've reached any end point, return the cost
        if (x, y) in end_points:
            return current_cost

        # Explore neighbors
        for nr, nc in get_neighbors(x, y, board_size):
            if (nr, nc) not in visited:
                new_cost = current_cost + cost_grid[nr, nc]
                heapq.heappush(pq, (new_cost, (nr, nc)))

    # If no path is found, return None
    return None

def weighted_shortest_path_feature(board_2d, board_size=7):
    red_cost = compute_weighted_shortest_path(board_2d, 'X', board_size)
    blue_cost = compute_weighted_shortest_path(board_2d, 'O', board_size)
    # Handle cases where a path might not exist
    if red_cost is None and blue_cost is None:
        return 'L'  # Default to Blue if no paths exist
    elif red_cost is None:
        return 'L'
    elif blue_cost is None:
        return 'K'
    return 'K' if red_cost < blue_cost else 'L'

def cluster_strength(board_2d, player, board_size=7):
    visited = set()
    
    def dfs(r, c):
        stack = [(r, c)]
        size = 0
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) not in visited and board_2d[rr, cc] == player:
                visited.add((rr, cc))
                size += 1
                for nr, nc in get_neighbors(rr, cc, board_size):
                    if board_2d[nr, nc] == player and (nr, nc) not in visited:
                        stack.append((nr, nc))
        return size

    component_sizes = []
    for r in range(board_size):
        for c in range(board_size):
            if board_2d[r, c] == player and (r, c) not in visited:
                comp_size = dfs(r, c)
                component_sizes.append(comp_size)
    
    return sum([cs**2 for cs in component_sizes])

def discretize_cluster_strength(red_cs, blue_cs):
    return 'I' if red_cs > blue_cs else 'J'

def longest_chain(board_2d, player, board_size=7):
    visited = set()
    max_chain_length = 0

    def dfs(row, col, length):
        nonlocal max_chain_length
        visited.add((row, col))
        max_chain_length = max(max_chain_length, length)
        for nr, nc in get_neighbors(row, col, board_size):
            if board_2d[nr, nc] == player and (nr, nc) not in visited:
                dfs(nr, nc, length + 1)

    for row in range(board_size):
        for col in range(board_size):
            if board_2d[row, col] == player and (row, col) not in visited:
                dfs(row, col, 1)

    return max_chain_length

def directional_dominance(board_2d, board_size=7):
    red_dominant_rows = 0
    blue_dominant_rows = 0
    for r in range(board_size):
        row_red = np.sum(board_2d[r, :] == 'X')
        row_blue = np.sum(board_2d[r, :] == 'O')
        if row_red > row_blue:
            red_dominant_rows += 1
        elif row_blue > row_red:
            blue_dominant_rows += 1

    red_dominant_cols = 0
    blue_dominant_cols = 0
    for c in range(board_size):
        col_red = np.sum(board_2d[:, c] == 'X')
        col_blue = np.sum(board_2d[:, c] == 'O')
        if col_red > col_blue:
            red_dominant_cols += 1
        elif col_blue > col_red:
            blue_dominant_cols += 1

    if red_dominant_rows > blue_dominant_cols:
        return 'N'  # Red directional dominance
    else:
        return 'M'  # Blue directional dominance or tie

def cluster_elongation_measure(board_2d, board_size=7):
    def largest_cluster_and_dims(player):
        visited = set()
        clusters = []
        for r in range(board_size):
            for c in range(board_size):
                if board_2d[r, c] == player and (r, c) not in visited:
                    stack = [(r, c)]
                    min_r, max_r = r, r
                    min_c, max_c = c, c
                    comp = []
                    while stack:
                        rr, cc = stack.pop()
                        if (rr, cc) not in visited and board_2d[rr, cc] == player:
                            visited.add((rr, cc))
                            comp.append((rr, cc))
                            min_r = min(min_r, rr)
                            max_r = max(max_r, rr)
                            min_c = min(min_c, cc)
                            max_c = max(max_c, cc)
                            for nr, nc in get_neighbors(rr, cc, board_size):
                                if board_2d[nr, nc] == player and (nr, nc) not in visited:
                                    stack.append((nr, nc))
                    clusters.append((comp, (max_r - min_r + 1, max_c - min_c + 1)))
        if not clusters:
            return (0, 1)  # No cluster, ratio=0
        largest = max(clusters, key=lambda x: len(x[0]))
        return largest[1]  # (height, width)

    red_dims = largest_cluster_and_dims('X')  # (height, width)
    blue_dims = largest_cluster_and_dims('O')

    # For Red (horizontal axis), elongation = width/height
    red_elong = red_dims[1] / red_dims[0] if red_dims[0] != 0 else 0
    # For Blue (vertical axis), elongation = height/width
    blue_elong = blue_dims[0] / blue_dims[1] if blue_dims[1] != 0 else 0

    return 'Q' if red_elong > blue_elong else 'P'

def calculate_balance_of_control(board, board_size=5):
    upper_half = board[: (board_size // 2) * board_size]
    lower_half = board[(board_size // 2) * board_size :]
    red_upper = np.sum(upper_half == 'X')
    blue_upper = np.sum(upper_half == 'O')
    red_lower = np.sum(lower_half == 'X')
    blue_lower = np.sum(lower_half == 'O')
    
    upper_diff = red_upper - blue_upper
    lower_diff = red_lower - blue_lower
    
    total_diff = upper_diff + lower_diff
    if total_diff > 0:
        return 'Y'  # Red controls more
    elif total_diff < 0:
        return 'Z'  # Blue controls more
    else:
        return 'Z'  # Balanced

def calculate_connectivity_score(board_2d, player, board_size=5):
    total_connections = 0
    piece_count = 0
    for r in range(board_size):
        for c in range(board_size):
            if board_2d[r, c] == player:
                piece_count += 1
                neighbors = get_neighbors(r, c, board_size)
                for nr, nc in neighbors:
                    if board_2d[nr, nc] == player:
                        total_connections += 1
    if piece_count == 0:
        return 0.0
    # Each connection is counted twice (for both pieces), so divide by 2
    average_connections = (total_connections / 2) / piece_count
    return average_connections

def calculate_critical_pieces(board, board_size=5):
    critical_red = 0
    critical_blue = 0
    for r in range(board_size):
        for c in range(board_size):
            idx = r * board_size + c
            piece = board[idx]
            if piece not in ['X', 'O']:
                continue
            neighbors = get_neighbors(r, c, board_size)
            for nr, nc in neighbors:
                neighbor_piece = board[nr * board_size + nc]
                if neighbor_piece != piece and neighbor_piece in ['X', 'O']:
                    if piece == 'X':
                        critical_red += 1
                    else:
                        critical_blue += 1
                    break  # Count each piece only once
    return 'T' if critical_red > critical_blue else 'U'

def longest_chain_feature(board_2d, board_size=7):
    red_chain_length = longest_chain(board_2d, 'X', board_size)
    blue_chain_length = longest_chain(board_2d, 'O', board_size)
    return 'R' if red_chain_length > blue_chain_length else 'S'

def calculate_cluster_strength(board_2d, board_size=7):
    red_cs = cluster_strength(board_2d, 'X', board_size)
    blue_cs = cluster_strength(board_2d, 'O', board_size)
    return red_cs, blue_cs

def extract_features(X_cat, board_size, FEATURE_SELECTION):
    features = []
    for board in X_cat:
        feature_dict = {}
        board_2d = board.reshape((board_size, board_size))
        if FEATURE_SELECTION[1]:  # ENABLE_MID_CTRL
            feature_dict['mid_control'] = calculate_mid_control_categorical(board, board_size)
        if FEATURE_SELECTION[2]:  # ENABLE_EDGE_PIECE_COUNT
            feature_dict['edge_piece_count'] = calculate_edge_piece_count(board, board_size)
        if FEATURE_SELECTION[3]:  # ENABLE_PIECE_COUNT
            feature_dict['piece_count'] = calculate_piece_count(board)
        if FEATURE_SELECTION[4]:  # ENABLE_NEIGHBOR_COUNT
            feature_dict['high_neighbor_count'] = calculate_high_neighbor_count(board, board_size)
        if FEATURE_SELECTION[5]:  # ENABLE_CLUSTER_STRENGTH
            red_cs, blue_cs = calculate_cluster_strength(board_2d, board_size)
            feature_dict['cluster_strength'] = discretize_cluster_strength(red_cs, blue_cs)
        if FEATURE_SELECTION[6]:  # ENABLE_DIRECTIONAL_DOMINANCE
            feature_dict['directional_dominance'] = directional_dominance(board_2d, board_size)
        if FEATURE_SELECTION[7]:  # ENABLE_CLUSTER_ELONGATION
            feature_dict['elongation_measure'] = cluster_elongation_measure(board_2d, board_size)
        if FEATURE_SELECTION[8]:  # ENABLE_LONGEST_CHAIN
            feature_dict['longest_chain'] = longest_chain_feature(board_2d, board_size)
        if FEATURE_SELECTION[9]:  # ENABLE_CRITICAL_PIECES
            feature_dict['critical_pieces'] = calculate_critical_pieces(board, board_size)
        if FEATURE_SELECTION[10]:  # ENABLE_CONNECTIVITY_SCORE
            score_red = calculate_connectivity_score(board_2d, 'X', board_size)
            score_blue = calculate_connectivity_score(board_2d, 'O', board_size)
            feature_dict['connectivity_score'] = 'V' if score_red > score_blue else 'W'  
        if FEATURE_SELECTION[11]:  # ENABLE_BALANCE_OF_CONTROL
            feature_dict['balance_of_control'] = calculate_balance_of_control(board, board_size)

        features.append(feature_dict)
    return features



def calculate_win_percentage(labels, player_label=0):
    total_games = len(labels)
    player_wins = np.sum(labels == player_label)
    return (player_wins / total_games) * 100


def convert_feature_values(features):
    converted_features = []
    for feature in features:
        converted_feature = {}
        for key, value in feature.items():
            if isinstance(value, str):
                # Simple binary encoding based on the feature's categorical values
                # Adjust this mapping based on specific feature requirements
                mapping = {
                    'A': 0, 'B': 1,
                    'C': 0, 'D': 1,
                    'E': 0, 'F': 1,
                    'G': 0, 'H': 1,
                    'I': 0, 'J': 1,
                    'K': 0, 'L': 1,
                    'M': 0, 'N': 1,
                    'P': 0, 'Q': 1,
                    'R': 0, 'S': 1,
                    'T': 0, 'U': 1,
                    'V': 0, 'W': 1,
                    'Y': 0, 'Z': 1
                }
                converted_feature[key] = mapping.get(value, 0)  # Default to 0 if undefined
            else:
                converted_feature[key] = value
        converted_features.append(converted_feature)
    return converted_features

def calculate_function_accuracy(converted_features, y_encoded):
    if not converted_features:
        return {}
    function_accuracy = {key: 0 for key in converted_features[0].keys()}
    
    for i, game in enumerate(converted_features[:len(y_encoded)]):
        for function, prediction in game.items():
            if prediction == y_encoded[i]:
                function_accuracy[function] += 1
    
    total_games = len(y_encoded)
    accuracy_percentage = {key: (correct / total_games) * 100 for key, correct in function_accuracy.items()}
    
    return accuracy_percentage

def print_function_accuracy(accuracy_percentage):
    print("\nAccuracy of each function at predicting the winner:")
    for function, acc in accuracy_percentage.items():
        print(f"{function}: {acc:.2f}%")
    print("\n")

# ----------------------------
# 4. Graph Construction
# ----------------------------

def generate_hex_adjacency(board_size=7):
    adjacency = [[] for _ in range(board_size * board_size)]
    for row in range(board_size):
        for col in range(board_size):
            node = row * board_size + col
            neighbors = []
            # Hex directions: up, down, left, right, up-right, down-left
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < board_size and 0 <= c < board_size:
                    neighbor = r * board_size + c
                    neighbors.append(neighbor)
            adjacency[node] = neighbors
    return adjacency

def encode_graphs(graphs, X_data, adjacency, board_size, FEATURE_SELECTION, labels, symbols, enable_cheat):
    '''
    Populates the Graphs object with node properties and edges.
    '''

    # Step 1: Set number of graph nodes
    for graph_id in range(X_data.shape[0]):
        graphs.set_number_of_graph_nodes(graph_id, board_size * board_size)

    # Step 2: Prepare node configuration
    graphs.prepare_node_configuration()

    # Step 3: Add graph nodes
    for graph_id in range(X_data.shape[0]):
        for node_id in range(board_size * board_size):
            total_edges = len(adjacency[node_id])
            graphs.add_graph_node(graph_id, node_id, total_edges)

    # Step 4: Prepare edge configuration
    graphs.prepare_edge_configuration()

    # Step 5: Extract and convert features
    features = extract_features(X_data, board_size, FEATURE_SELECTION)
    converted_features = convert_feature_values(features)

    # Step 6: Add node properties
    for graph_id in range(X_data.shape[0]):
        for node_id in range(board_size * board_size):
            # Add the basic symbol property
            sym = X_data[graph_id][node_id]
            #print(f"sym: {sym}, added to Node-{node_id}, of Graph-{graph_id}")
            graphs.add_graph_node_property(graph_id, node_id, sym)
            
            # Add cheat feature if enabled
            if enable_cheat and FEATURE_SELECTION[0]:  # ENABLE_CHEAT
                graphs.add_graph_node_property(graph_id, node_id, str(labels[graph_id]))
            
            # Add categorical features using original symbols
            if FEATURE_SELECTION[1]:  # ENABLE_MID_CTRL
                feature_val = features[graph_id].get("mid_control", 'A')  # Default to 'A'
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[2]:  # ENABLE_EDGE_PIECE_COUNT
                feature_val = features[graph_id].get("edge_piece_count", 'C') 
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[3]:  # ENABLE_PIECE_COUNT
                feature_val = features[graph_id].get("piece_count", 'E')  
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[4]:  # ENABLE_NEIGHBOR_COUNT
                feature_val = features[graph_id].get("high_neighbor_count", 'G') 
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[5]:  # ENABLE_CLUSTER_STRENGTH
                feature_val = features[graph_id].get("cluster_strength", 'I')  
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[6]:  # ENABLE_DIRECTIONAL_DOMINANCE
                feature_val = features[graph_id].get("directional_dominance", 'K')  
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[7]:  # ENABLE_CLUSTER_ELONGATION
                feature_val = features[graph_id].get("elongation_measure", 'M')  
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[8]:  # ENABLE_LONGEST_CHAIN
                feature_val = features[graph_id].get("longest_chain", 'P')  
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[9]:  # ENABLE_CRITICAL_PIECES
                feature_val = features[graph_id].get("critical_pieces", 'R')  
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[10]:  # ENABLE_CONNECTIVITY_SCORE
                feature_val = features[graph_id].get("connectivity_score", 'T')  
                graphs.add_graph_node_property(graph_id, node_id, feature_val)
            
            if FEATURE_SELECTION[11]:  # ENABLE_BALANCE_OF_CONTROL
                feature_val = features[graph_id].get("balance_of_control", 'V') 
                graphs.add_graph_node_property(graph_id, node_id, feature_val)


    # Step 7: Add edges with or without additional edge types
    if ADDITIONAL_EDGES:
        for graph_id in range(X_data.shape[0]):
            for node_id, neighbors in enumerate(adjacency):
                for neighbor in neighbors:
                    if X_data[graph_id][node_id] == X_data[graph_id][neighbor]:
                        if X_data[graph_id][node_id] == 'X':
                            edge_type = "BuddyX"
                        elif X_data[graph_id][node_id] == 'O':
                            edge_type = "BuddyO"
                        elif X_data[graph_id][node_id] == ' ':
                            edge_type = "Empty Buddy"
                        else:
                            edge_type = "Not Buddy"
                    else:
                        edge_type = "Not Buddy"
                    graphs.add_graph_node_edge(graph_id, node_id, neighbor, edge_type)
    else:
        for graph_id in range(X_data.shape[0]):
            for node_id, neighbors in enumerate(adjacency):
                for neighbor in neighbors:
                    edge_type = "Connected" if X_data[graph_id][node_id] == X_data[graph_id][neighbor] else "Not Connected"
                    graphs.add_graph_node_edge(graph_id, node_id, neighbor, edge_type)

# ----------------------------
# Hyperparameter Tuning
# ----------------------------

def tune_s_parameter(s_initial, max_trials, epochs_per_trial, increment, number_of_clauses, depth, message_size, message_bits, max_included_literals, graphs_train, y_train, graphs_val, y_val, logging):
    '''
    Tunes the 's' parameter to find the best validation accuracy.
    '''
    overall_best_s = s_initial
    overall_best_val_accuracy = 0
    train_accuracies_s = []
    val_accuracies_s = []
    s_values = []
    
    for trial in range(max_trials):
        s = s_initial + (trial * increment)
        
        tm = MultiClassGraphTsetlinMachine(
            number_of_clauses=number_of_clauses,
            T=round(number_of_clauses / 1.6),
            s=s,
            depth=depth,
            message_size=message_size,
            message_bits=message_bits,
            max_included_literals=max_included_literals,
            grid=(16 * 13, 1, 1),  
            block=(128, 1, 1)       
        )
        
        logging.info(f"\nBegin tuning iteration #{trial + 1}....")
        logging.info(f"s value: {s}")
        start_training = time.time()
        
        current_best_val_accuracy = 0
        current_best_train_accuracy = 0
        
        for epoch in range(epochs_per_trial):
            trial_time = time.time()
            tm.fit(graphs_train, y_train, epochs=1, incremental=True)
    
            train_preds = tm.predict(graphs_train)
            train_accuracy = np.mean(train_preds == y_train)
        
            val_preds = tm.predict(graphs_val)
            val_accuracy = np.mean(val_preds == y_val)
    
            # Update current best accuracies for this `s`
            if train_accuracy > current_best_train_accuracy:
                current_best_train_accuracy = train_accuracy
    
            if val_accuracy > current_best_val_accuracy:
                current_best_val_accuracy = val_accuracy
    
            # Update overall best accuracy and best `s` value across all iterations
            if val_accuracy > overall_best_val_accuracy:
                overall_best_val_accuracy = val_accuracy
                overall_best_s = s
    
            logging.info(f"Trial epoch#{epoch + 1} -- Accuracy train: {train_accuracy * 100:.2f}% -- Accuracy val: {val_accuracy * 100:.2f}% -- Duration: {time.time() - trial_time:.2f} seconds.")
                
        train_accuracies_s.append(current_best_train_accuracy)
        val_accuracies_s.append(current_best_val_accuracy)
        s_values.append(s)
    
        logging.info(f"T: {round(number_of_clauses / 1.6)}, s: {s}, Best val acc: {current_best_val_accuracy:.3f}.")
        logging.info(f"Best s value trial run: {overall_best_s} at iteration {trial + 1}.")
        logging.info(f"Total time for trial: {time.time() - start_training:.2f} seconds.\n")
    
    logging.info(f"\nOverall best s value: {overall_best_s} with val accuracy: {overall_best_val_accuracy:.3f}")
    
    # Plot Training and Validation Accuracies vs s Values
    plt.figure(figsize=(14, 8))
    plt.plot(s_values, [acc * 100 for acc in train_accuracies_s], marker='o', linestyle='-', label='Training Accuracy', color='blue')
    plt.plot(s_values, [acc * 100 for acc in val_accuracies_s], marker='o', linestyle='--', label='Validation Accuracy', color='purple')
    plt.title('Training and Validation Accuracies vs s Values')
    plt.xlabel('s values')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to a file
    filename = f'Accuracies_vs_s_BestValAcc-{overall_best_val_accuracy:.2f}_s-{overall_best_s}.png'
    plt.savefig(filename)
    logging.info(f"Plot saved as {filename}")
    plt.close()
    
    return overall_best_s, overall_best_val_accuracy





# ----------------------------
#  Training the Model
# ----------------------------



def train_model(tm, graphs_train, y_train, graphs_val, y_val, epochs, logging):
    '''
    Trains the Graph Tsetlin Machine and logs metrics.
    '''
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    logging.info("Starting training...")
    start_training = time.time()
    for epoch in range(epochs):
        print("")
        logging.info(f"\tEpoch {epoch + 1}/{epochs}: ")
        
        # Record training start time
        epoch_start = time.time()
        
        # Train for one epoch
        tm.fit(graphs_train, y_train, epochs=1, incremental=True)
    
        # Predict on training data
        train_preds = tm.predict(graphs_train)
        train_accuracy = np.mean(train_preds == y_train)
        train_accuracies.append(train_accuracy)
        
        # Predict on validation data
        val_preds = tm.predict(graphs_val)
        val_accuracy = np.mean(val_preds == y_val)
        val_accuracies.append(val_accuracy)
    
        # Calculate loss 
        train_loss = np.mean((y_train - train_preds) ** 2)
        val_loss = np.mean((y_val - val_preds) ** 2)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
        training_time = time.time() - epoch_start
        logging.info(f"\tTrain Accuracy: \t\t{train_accuracy * 100:.2f}%")
        logging.info(f"\tTrain Loss: \t\t\t{train_loss:.5f}")
        logging.info(f"\tValidation Accuracy: \t\t{val_accuracy * 100:.2f}%")
        logging.info(f"\tValidation Loss: \t\t{val_loss:.5f}")
        logging.info(f"\tSample Train Predictions:\t{train_preds[:20]}, Labels: {y_train[:20]}")
        logging.info(f"\tSample Val Predictions:\t\t{val_preds[:20]}, Labels: {y_val[:20]}")
        logging.info(f"\tTraining Time:\t\t\t{training_time:.2f} seconds.")
    
    total_training_time = time.time() - start_training
    logging.info(f"\nTotal Training Time: {total_training_time:.2f} seconds.")
    
    return train_accuracies, val_accuracies, train_losses, val_losses




# ----------------------------
# Evaluation and Visualization
# ----------------------------


def plot_board(game, board_size, true_label, pred_label, index):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)
    ax.set_xticks(range(board_size))
    ax.set_yticks(range(board_size))
    ax.grid(True)
    ax.set_aspect('equal')
    for row in range(board_size):
        for col in range(board_size):
            cell = game[row * board_size + col]
            if cell == 'X':
                ax.add_patch(patches.Circle((col + 0.5, board_size - row - 0.5), 0.3, color='red'))
            elif cell == 'O':
                ax.add_patch(patches.Circle((col + 0.5, board_size - row - 0.5), 0.3, color='blue'))
    plt.title(f"Game Index: {index}\nTrue: {'Red' if true_label == 0 else 'Blue'}, Predicted: {'Red' if pred_label == 0 else 'Blue'}")
    plt.savefig(f'{folder_name}/misclassified_game_{index}.png')
    plt.close()


def evaluate_model(tm, graphs, y_true, X_data, data_set_type, board_size, logging):
    '''
    Evaluates the trained model on a given set, logs the results, and prints misclassified games.
    '''
    preds = tm.predict(graphs)
    accuracy = np.mean(preds == y_true)
    
    logging.info(f"{data_set_type} Accuracy: {accuracy * 100:.2f}%")
    
    # Classification Report
    logging.info(f"{data_set_type} Classification Report:")
    logging.info("\n" + classification_report(y_true, preds))
    
    # Confusion Matrix
    logging.info(f"Confusion Matrix ({data_set_type} Set):")
    cm = confusion_matrix(y_true, preds)
    logging.info(f"\n{cm}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({data_set_type} Set)')
    plt.colorbar()
    classes = ['Red Wins', 'Blue Wins']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Normalize the confusion matrix.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]} ({cm_normalized[i, j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt_filename = f'{folder_name}/confusion_matrix_Epochs-{EPOCHS}_Version-{VERSION}_{TIMESTAMP}_{data_set_type.lower()}_set.png'
    plt.savefig(plt_filename)
    logging.info(f"Confusion matrix plot saved as '{plt_filename}'")
    plt.close()
    
    # Identify Misclassified Games
    misclassified_indices = np.where(preds != y_true)[0]
    misclassified_games = X_data[misclassified_indices]
    misclassified_true_labels = y_true[misclassified_indices]
    misclassified_pred_labels = preds[misclassified_indices]
    
    # Print Misclassified Games
    print(f"\n--- First 5 Misclassified Games ({data_set_type} Set) ---\n")
    for idx in misclassified_indices[:5]:  # Limit to the first 5 misclassified indices
        game = X_data[idx]
        true_label = y_true[idx]
        pred_label = preds[idx]
        print(f"Game Index: {idx}")
        print(f"Board State:")
        for row in range(board_size):
            row_state = ' '.join(game[row * board_size:(row + 1) * board_size])
            print(row_state)
        print(f"True Winner: {'Red' if true_label == 0 else 'Blue'}")
        print(f"Predicted Winner: {'Red' if pred_label == 0 else 'Blue'}")
        print("-----------------------------------\n")
    
    # After identifying misclassified games, plot the first 5 only
    for idx in misclassified_indices[:5]:  # Limit to the first 5 misclassified indices
        game = X_data[idx]
        true_label = y_true[idx]
        pred_label = preds[idx]
        plot_board(game, board_size, true_label, pred_label, idx)

    
    # Optionally, save misclassified games to a file
    misclassified_data = []
    for idx in misclassified_indices:
        game = X_data[idx]
        game_dict = {f'cell{row}_{col}': game[row * board_size + col] 
                    for row in range(board_size) for col in range(board_size)}
        game_dict['true_winner'] = 'Red' if y_true[idx] == 0 else 'Blue'
        game_dict['predicted_winner'] = 'Red' if preds[idx] == 0 else 'Blue'
        misclassified_data.append(game_dict)
    
    misclassified_df = pd.DataFrame(misclassified_data)
    misclassified_df.to_csv(f'{folder_name}/misclassified_games_{data_set_type.lower()}_set.csv', index=False)
    logging.info(f"Misclassified games saved as 'misclassified_games_{data_set_type.lower()}_set.csv'")
    
    return preds


def plot_training_metrics(train_accuracies, val_accuracies, train_losses, val_losses, epochs, version, data_file_path, timestamp, logging):
    # Plot Accuracies
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), [acc * 100 for acc in train_accuracies], label='Train Accuracy')
    plt.plot(range(1, epochs + 1), [acc * 100 for acc in val_accuracies], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training and Validation Accuracy Over Epochs, Version: {VERSION}\nDataset: {DATA_FILE_PATH}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    accuracy_plot_filename = f'{folder_name}/overall_accuracy_Epochs-{EPOCHS}_Version-{VERSION}_{TIMESTAMP}.png'
    plt.savefig(accuracy_plot_filename)
    logging.info(f"Accuracy plot saved as {accuracy_plot_filename}")
    plt.close()
    
    # Plot Losses
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Over Epochs, Version: {VERSION}\nDataset: {DATA_FILE_PATH}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_filename = f'{folder_name}/lloss_plot_Epochs-{EPOCHS}_Version-{VERSION}_{TIMESTAMP}.png'
    plt.savefig(loss_plot_filename)
    logging.info(f"Loss plot saved as {loss_plot_filename}")
    plt.close()



def interpret_clause(expression, feature_mapping):
    '''
    Translates a clause expression into a concise, human-readable format.
    '''
    literals = expression.split(' AND ')
    interpreted_literals = []
    
    for lit in literals:
        lit = lit.strip()
        is_negated = False
        if lit.startswith('NOT '):
            is_negated = True
            lit = lit[4:]
        
        feature_info = feature_mapping.get(lit, None)
        if feature_info and feature_info['feature'] != 'Unknown':
            node_id = feature_info['node_id']
            feature = feature_info['feature']
            # Format: Node-<node_id>_<feature>
            formatted_literal = f"Node-{node_id}_{feature}"
            if is_negated:
                formatted_literal = f"NOT {formatted_literal}"
            interpreted_literals.append(formatted_literal)
        else:
            # Handle unmapped or unknown literals
            formatted_literal = f"Unknown_Literal({lit})"
            if is_negated:
                formatted_literal = f"NOT {formatted_literal}"
            interpreted_literals.append(formatted_literal)
            
            # Optionally, log the unmapped literal
            print(f"Warning: Literal '{lit}' is not mapped to any feature.")
    
    return " AND ".join(interpreted_literals)


    
def generate_feature_mapping(symbols, board_size, X_data):
    feature_mapping = {}
    number_of_features_per_node = len(symbols)
    number_of_nodes = board_size * board_size
    
    # Determine total possible literals based on data
    max_literal = X_data.shape[1] * number_of_features_per_node
    
    for literal_num in range(max_literal):
        lit = f'x{literal_num}'
        node_id = literal_num // number_of_features_per_node
        feature_idx = literal_num % number_of_features_per_node
        if node_id < number_of_nodes and feature_idx < len(symbols):
            feature = symbols[feature_idx]
            feature_mapping[lit] = {
                'node_id': node_id,
                'position': (node_id // board_size, node_id % board_size),
                'feature': feature
            }
        else:
            # Handle literals beyond the expected range
            feature_mapping[lit] = {
                'node_id': node_id,
                'position': (node_id // board_size, node_id % board_size),
                'feature': 'Unknown'
            }
    
    # Save to CSV for reference
    mapping_df = pd.DataFrame([
        {'Literal': lit, 'Node_ID': details['node_id'], 
         'Position': details['position'], 'Feature': details['feature']}
        for lit, details in feature_mapping.items()
    ])
    
    mapping_df.to_csv('feature_mapping.csv', index=False)
    print("Feature mapping saved to 'feature_mapping.csv'")
    
    return feature_mapping




# ----------------------------
# Main Execution Flow
# ----------------------------

def main():
    
    X_cat, y_encoded = load_and_prepare_data(
        data_file_path=DATA_FILE_PATH,
        board_size=BOARD_SIZE,
        data_reduction_factor=DATA_REDUCTION_FACTOR,
        random_state=RANDOM_STATE
    )

    # Generate feature mapping before splitting
    logging.info("Generate feature mapping...")
    feature_mapping = generate_feature_mapping(symbols, BOARD_SIZE, X_cat)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_cat, y_encoded, 
        test_size=VAL_TEST_SPLIT_RATIO, 
        random_state=RANDOM_STATE, 
        stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=RANDOM_STATE
    )
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Calculate win percentages before preprocessing
    train_win_percentage = calculate_win_percentage(y_train, player_label=0)
    val_win_percentage = calculate_win_percentage(y_val, player_label=0)
    test_win_percentage = calculate_win_percentage(y_test, player_label=0)
    
    # Verify class distribution
    print(f"\nTraining label distribution: {Counter(y_train)}")  
    print(f"Validation label distribution: {Counter(y_val)}")  
    print(f"Testing label distribution: {Counter(y_test)}")  
    
    print(f"\nPlayer 1 Win Percentage in Training Set: {train_win_percentage:.2f}%")
    print(f"Player 1 Win Percentage in Validation Set: {val_win_percentage:.2f}%")
    print(f"Player 1 Win Percentage in Testing Set: {test_win_percentage:.2f}%\n")

    
    # ----------------------------
    # Feature Extraction
    # ----------------------------
    
    logging.info("Extracting features...")
    features = extract_features(X_cat, BOARD_SIZE, FEATURE_SELECTION)
    converted_features = convert_feature_values(features)
    
    # Calculate and print function accuracies
    accuracy_percentage = calculate_function_accuracy(converted_features, y_encoded)
    print_function_accuracy(accuracy_percentage)
    
    # ----------------------------
    # Graph Construction and Encoding
    # ----------------------------
    
    adjacency = generate_hex_adjacency(BOARD_SIZE)
    
    # Verify adjacency for the first few nodes
    for node_id in range(min(BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)):
        print(f"Node {node_id} neighbors: {adjacency[node_id]}")
    
    # Initialize Graphs objects
    graphs_train = Graphs(
        number_of_graphs=X_train.shape[0],
        symbols=symbols,
        hypervector_size=HYPERVECTOR_SIZE,
        hypervector_bits=HYPERVECTOR_BITS,
        double_hashing=DOUBLE_HASHING
    )
    
    graphs_val = Graphs(
        number_of_graphs=X_val.shape[0],
        init_with=graphs_train
    )
    
    graphs_test = Graphs(
        number_of_graphs=X_test.shape[0], 
        init_with=graphs_train
    )
    
    # Encode graphs
    logging.info("Encoding training graphs...")
    encode_graphs(graphs_train, X_train, adjacency, BOARD_SIZE, FEATURE_SELECTION, y_train, symbols, ENABLE_CHEAT)
    logging.info("Encoding validation graphs...")
    encode_graphs(graphs_val, X_val, adjacency, BOARD_SIZE, FEATURE_SELECTION, y_val, symbols, ENABLE_CHEAT)
    logging.info("Encoding testing graphs...")
    encode_graphs(graphs_test, X_test, adjacency, BOARD_SIZE, FEATURE_SELECTION, y_test, symbols, ENABLE_CHEAT)
    
    # Finalize encoding
    logging.info("Finalizing encoding for graphs...")
    graphs_train.encode()
    graphs_val.encode()
    graphs_test.encode()
    
    # ----------------------------
    # Hyperparameter Tuning (Optional)
    # ----------------------------
    
    overall_best_s = S_INITIAL
    if ENABLE_TUNING_S:
        logging.info("Starting hyperparameter tuning for 's'...")
        overall_best_s, overall_best_val_accuracy = tune_s_parameter(
            s_initial=S_INITIAL,
            max_trials=MAX_TRIALS,
            epochs_per_trial=EPOCHS_PER_TRIAL,
            increment=INCREMENT_S_PER_TRIAL_BY,
            number_of_clauses=NUMBER_OF_CLAUSES,
            depth=DEPTH,
            message_size=MESSAGE_SIZE,
            message_bits=MESSAGE_BITS,
            max_included_literals=MAX_INCLUDED_LITERALS,
            graphs_train=graphs_train,
            y_train=y_train,
            graphs_val=graphs_val,
            y_val=y_val,
            logging=logging
        )
    else:
        overall_best_s = S_INITIAL
        logging.info(f"Using initial 's' value: {overall_best_s}")
    
    # ----------------------------
    # Initialize the Graph Tsetlin Machine
    # ----------------------------
    
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=NUMBER_OF_CLAUSES,
        T=T,
        s=overall_best_s,
        depth=DEPTH,
        message_size=MESSAGE_SIZE,
        message_bits=MESSAGE_BITS,
        max_included_literals=MAX_INCLUDED_LITERALS,
        grid=(16 * 13, 1, 1),   # Adjust based on system capabilities
        block=(128, 1, 1)        # Adjust based on system capabilities
    )
    
    # ----------------------------
    # Training the Model
    # ----------------------------
    
    train_accuracies, val_accuracies, train_losses, val_losses = train_model(
        tm=tm,
        graphs_train=graphs_train,
        y_train=y_train,
        graphs_val=graphs_val,
        y_val=y_val,
        epochs=EPOCHS,
        logging=logging
    )
    
    # ----------------------------
    # Evaluation
    # ----------------------------
    
    # Evaluate on Training Set
    train_preds = evaluate_model(
        tm=tm,
        graphs=graphs_train,
        y_true=y_train,
        X_data=X_train,
        data_set_type='Training',
        board_size=BOARD_SIZE,
        logging=logging
    )
    
    # Evaluate on Validation Set
    val_preds = evaluate_model(
        tm=tm,
        graphs=graphs_val,
        y_true=y_val,
        X_data=X_val,
        data_set_type='Validation',
        board_size=BOARD_SIZE,
        logging=logging
    )
    
    # Evaluate on Testing Set
    test_preds = evaluate_model(
        tm=tm,
        graphs=graphs_test,
        y_true=y_test,
        X_data=X_test,
        data_set_type='Testing',
        board_size=BOARD_SIZE,
        logging=logging
    )
    
    # ----------------------------
    # Visualization
    # ----------------------------
    
    plot_training_metrics(
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        train_losses=train_losses,
        val_losses=val_losses,
        epochs=EPOCHS,
        version=VERSION,
        data_file_path=DATA_FILE_PATH,
        timestamp=timestamp,
        logging=logging
    )


    
    # After training and retrieving model clauses
    weights = tm.get_state()[1].reshape(2, -1)  # Assuming two classes
    
    # Calculate the absolute weight sum for sorting
    weight_sums = weights[0]  # Considering Red weights
    top_n = 5
    top_clause_indices = np.argsort(weight_sums)[-top_n:][::-1]  # Top N clauses
    
    print("\n--- Top 5 Clauses by Red Weight ---\n")
    for clause_idx in top_clause_indices:
        literals = []
        for hv_idx in range(HYPERVECTOR_SIZE * 2):
            if tm.ta_action(0, clause_idx, hv_idx):
                if hv_idx < HYPERVECTOR_SIZE:
                    literals.append(f"x{hv_idx}")               # Positive literal
                else:
                    literals.append(f"NOT x{hv_idx - HYPERVECTOR_SIZE}")  # Negative literal
        clause_expression = " AND ".join(literals)
        if len(literals) != 0: 
            logging.info(f"Clause #{clause_idx + 1}:")
            logging.info(f"  Weights: (Red: {weights[0, clause_idx]}, Blue: {weights[1, clause_idx]})")
            logging.info(f"  Expression: {clause_expression}")
            logging.info(f"  Number of Literals: {len(literals)}\n")
            
            # Interpret the clause using the concise function
            interpreted_expr = interpret_clause(clause_expression, feature_mapping)
            logging.info(f"  Interpreted Expression: {interpreted_expr}\n")


    
        # Calculate the absolute weight sum for sorting
    weight_sums = weights[1]  # Considering Blue weights
    top_n = 5
    top_clause_indices = np.argsort(weight_sums)[-top_n:][::-1]  # Top N clauses

    print("\n--- Top 5 Clauses by Blue Weight ---\n")
    for clause_idx in top_clause_indices:
        literals = []
        for hv_idx in range(HYPERVECTOR_SIZE * 2):
            if tm.ta_action(0, clause_idx, hv_idx):
                if hv_idx < HYPERVECTOR_SIZE:
                    literals.append(f"x{hv_idx}")               # Positive literal
                else:
                    literals.append(f"NOT x{hv_idx - HYPERVECTOR_SIZE}")  # Negative literal
        clause_expression = " AND ".join(literals)
        if len(literals) != 0: 
            logging.info(f"Clause #{clause_idx + 1}:")
            logging.info(f"  Weights: (Red: {weights[0, clause_idx]}, Blue: {weights[1, clause_idx]})")
            logging.info(f"  Expression: {clause_expression}")
            logging.info(f"  Number of Literals: {len(literals)}")
            interpreted_expr = interpret_clause(clause_expression, feature_mapping)
            logging.info(f"  Interpreted Expression: {interpreted_expr}\n")
            

        # Initialize counter for non-empty clauses
        total_non_empty_clauses = 0
        
        # Iterate through all clauses to count non-empty ones
        for clause_idx in range(NUMBER_OF_CLAUSES):
            literals = []
            for hv_idx in range(HYPERVECTOR_SIZE * 2):
                if tm.ta_action(0, clause_idx, hv_idx):
                    if hv_idx < HYPERVECTOR_SIZE:
                        literals.append(f"x{hv_idx}")  # Positive literal
                    else:
                        literals.append(f"NOT x{hv_idx - HYPERVECTOR_SIZE}")  # Negative literal
            if len(literals) > 0:
                total_non_empty_clauses += 1
        

    logging.info(f"Total non-empty clauses: {total_non_empty_clauses} out of {NUMBER_OF_CLAUSES}")
    total_features = BOARD_SIZE * BOARD_SIZE * len(symbols)
    logging.info(f"Total features (literals): {total_features}\n\n")

    # Classification Report and Confusion Matrix
    logging.info("Training Classification Report:")
    logging.info("\n" + classification_report(y_train, train_preds))
    
    logging.info("Validation Classification Report:")
    logging.info("\n" + classification_report(y_val, val_preds))
    
    logging.info("Testing Classification Report:")
    logging.info("\n" + classification_report(y_test, test_preds))
    
    logging.info("Confusion Matrix (Test Set):")
    logging.info(confusion_matrix(y_test, test_preds))
    
    logging.info("Training and evaluation completed successfully.")




if __name__ == "__main__":
    main()
