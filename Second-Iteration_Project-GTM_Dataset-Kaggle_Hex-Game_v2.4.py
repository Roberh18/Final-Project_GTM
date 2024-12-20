import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import resample
import sys
import os
import logging
import datetime


# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

'''
Threshold (T):
T controls the number of clauses required to vote for a particular decision during classification. In essence, it acts as a threshold for aggregating the votes of the clauses.
- A higher T value makes the model more conservative in its predictions. The model will require more clauses to agree before making a decision, leading to potentially lower sensitivity but higher specificity.
- A lower T value makes the model more permissive, enabling quicker but potentially noisier decisions.

Specificity Parameter (s):
The s parameter controls the bias of the Tsetlin Automaton towards including or excluding literals in the clauses. It affects how specific or general the clauses are.
- A higher s value biases the clauses towards specificity, making them more focused and precise in their coverage. This can lead to overfitting if s is too high.
- A lower s value results in more general clauses, which may help in generalization but might fail to capture specific patterns, leading to underfitting.

'''

VERSION = "2.4"
ENABLE_TUNING_s = False
MAX_TRIALS = 25                    # 100
EPOCHS_PER_TRIAL = 30               # 20
Increment_s_pr_trial_by = 0.5
epochs = 25                       # Number of training epochs 200                        
number_of_clauses = 20000           # Number of clauses
T = number_of_clauses * 1.6        # Threshold (0.625)
s = 0.05                             # Specificity parameter [0.03-2]
depth = 5                          # Depth of the machine
hypervector_size = 512             # Size of hypervectors
hypervector_bits = 2               # Bits per hypervector element
message_size = 512                # Size of the message space
message_bits = 2                   # Bits per message element
double_hashing = False             # Double hashing option
max_included_literals = 32         # Maximum literals in a clause

# Game-specific parameters        # Symbols for the board ('X': Red, 'O': Blue, '-': Empty), P0 RED mid ctrl, P1 BLUE,
board_size = 7                     # Size of the Hex board
data_reduction_factor = 0.01       # Fraction of the dataset to keep

# Split data into training and testing sets with stratification to maintain class balance
val_test_split_ratio = 0.3  # 0.3, for 15% for validation and 15% test split
random_state = 42      # For reproducibility

# FEATURE SELECTION
ENABLE_CHEAT = False                     # Overpowered 
ENABLE_MID_CTRL = True                  #  59%
ENABLE_CORNER_CONTROL = True            #  56%
ENABLE_EDGE_PIECE_COUNT = True          #  56%
ENABLE_PIECE_COUNT = False               # Overpowered
ENABLE_NEIGHBOR_COUNT = True            #  58%
ENABLE_MINIMAL_PATH_LENGTH = False        # Overpowered 
ENABLE_EDGE_TO_EDGE_DISTANCE = False      # Overpowered 
ENABLE_BRIDGE_CONNECTIONS = False         #  Overpowered
ENABLE_CLUSTER_STRENGTH = True           #  83%



# Create a dynamic log file name with datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join("logs", f"training_log_epochs-{epochs}SW-{VERSION}_{timestamp}.log")

# Configure logging
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




FEATURE_SELECTION = [
    ENABLE_CHEAT, 
    ENABLE_MID_CTRL, 
    ENABLE_CORNER_CONTROL, 
    ENABLE_EDGE_PIECE_COUNT, 
    ENABLE_PIECE_COUNT, 
    ENABLE_NEIGHBOR_COUNT, 
    ENABLE_MINIMAL_PATH_LENGTH,
    ENABLE_EDGE_TO_EDGE_DISTANCE,
    ENABLE_BRIDGE_CONNECTIONS,
    ENABLE_CLUSTER_STRENGTH
]


# Define feature-symbol mappings
feature_symbols = {
    0: ['0', '1'],               # ENABLE_CHEAT
    1: ['A', 'B'],               # ENABLE_MID_CTRL
    2: ['C', 'D'],               # ENABLE_CORNER_CONTROL
    3: ['E', 'F'],               # ENABLE_EDGE_PIECE_COUNT
    4: ['G', 'H'],               # ENABLE_PIECE_COUNT
    5: ['I', 'J'],               # ENABLE_NEIGHBOR_COUNT
    6: ['K', 'L'],               # ENABLE_MINIMAL_PATH_LENGTH
    7: ['M', 'N'],               # ENABLE_EDGE_TO_EDGE_DISTANCE
    8: ['P', 'Q'],               # ENABLE_BRIDGE_CONNECTIONS
    9: ['R', 'S'],               # ENABLE_CLUSTER_STRENGTH
}

symbols = ['X', 'O', ' ']

# Add additional symbols based on enabled features
logging.info("\n\nBuilding list of symbols...")
for index, is_enabled in enumerate(FEATURE_SELECTION):
    if is_enabled:
        symbols.extend(feature_symbols[index])

symbols.extend(["BuddyX", "BuddyO", "Empty Buddy", "Not Buddy"])
       
print("\n")

# Log hyperparameters and setup info
logging.info("Starting training run...")
logging.info(f"VERSION: {VERSION}")
logging.info("Hyperparameters and configuration:")
logging.info(f"epochs = {epochs}")
logging.info(f"number_of_clauses = {number_of_clauses}")
logging.info(f"T = {T}")
logging.info(f"s = {s}")
logging.info(f"depth = {depth}")
logging.info(f"hypervector_size = {hypervector_size}")
logging.info(f"hypervector_bits = {hypervector_bits}")
logging.info(f"message_size = {message_size}")
logging.info(f"message_bits = {message_bits}")
logging.info(f"double_hashing = {double_hashing}")
logging.info(f"max_included_literals = {max_included_literals}")
logging.info(f"board_size = {board_size}")
logging.info(f"data_reduction_factor = {data_reduction_factor}")
logging.info(f"val_test_split_ratio = {val_test_split_ratio}")
logging.info(f"random_state = {random_state}")
logging.info(f"FEATURE_SELECTION = {FEATURE_SELECTION}")
logging.info(f"Symbols = {symbols}")
print("\n")




# ----------------------------
# 2. Data Preparation
# ----------------------------


data_file_path = 'hex_games_1_000_000_size_7.csv'

# Load data
try:
    data = pd.read_csv(data_file_path)
    print(f"Data loaded successfully from {data_file_path}.")
except FileNotFoundError:
    print(f"Error: The file {data_file_path} was not found.")
    sys.exit(1)

# Reduce dataset size
total_samples = len(data)
reduced_samples = int(total_samples * data_reduction_factor)
data = data.sample(n=reduced_samples, random_state=42).reset_index(drop=True)
print(f"Dataset reduced to {reduced_samples} samples.")

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
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

# Extract feature columns (cell0_0 to cell6_6)
feature_columns = [f'cell{row}_{col}' for row in range(board_size) for col in range(board_size)]
X = data[feature_columns].values  # Features (Shape: num_samples, 49)
y = data['winner'].values         # Labels (Shape: num_samples,)

logging.info(f"Total samples: {X.shape[0]}")
logging.info(f"Feature shape: {X.shape[1]}")  # Should be 49 for 7x7 board

# Debug: Check raw numeric values
logging.info(f"Unique values in raw data: {np.unique(X)}")

# Encode features
def encode_features(X_numeric):
    """
    Convert numerical cell values to categorical symbols.
    Mapping:
        1  -> 'X' (Red)
        -1 -> 'O' (Blue)
        0  -> ' ' (Empty)
    """
    symbol_mapping = {1: 'X', -1: 'O', 0: ' '}
    X_categorical = np.vectorize(symbol_mapping.get)(X_numeric).astype(str)  # Force string type
    # Debugging: Ensure no invalid symbols remain
    unique_symbols = np.unique(X_categorical)
    print(f"Unique symbols after encoding: {unique_symbols}")
    return X_categorical

X_cat = encode_features(X)

# Debug: Verify encoded values
logging.info(f"X_cat after encoding: \n{X_cat[:1]}\n")

# Encode labels
def encode_labels(y):
    """
    Convert winner labels from 1 and -1 to 0 and 1.
    Mapping:
        1  -> 0 (Red Wins)
        -1 -> 1 (Blue Wins)
    """
    label_mapping = {1: 0, -1: 1}  # Red:0, Blue:1
    y_encoded = np.vectorize(label_mapping.get)(y)
    return y_encoded

y_encoded = encode_labels(y)






# ----------------------------
# 3. FEATURE extraction
# ----------------------------

def calculate_mid_control_categorical(board, board_size=7):
    board_2d = board.reshape((board_size, board_size))
    mid_matrix = board_2d[2:5, 2:5]
    red_count = np.sum(mid_matrix == 'X')  # Count Red ('X')
    blue_count = np.sum(mid_matrix == 'O')  # Count Blue ('O')
    if red_count > blue_count:
        return 'A'  # Red controls
    else:
        return 'B'  # Blue controls

def calculate_corner_control(board, board_size=7):
    corners = [0, board_size - 1, board_size * (board_size - 1), board_size**2 - 1]
    red_count = sum(1 for corner in corners if board[corner] == 'X')
    blue_count = sum(1 for corner in corners if board[corner] == 'O')
    if red_count > blue_count:
        return 'C'  # Red controls
    else:
        return 'D'  # Blue controls

def calculate_edge_piece_count(board, board_size=7):
    edge_indices = set()
    edge_indices.update(range(board_size))  # Top
    edge_indices.update(range(board_size * (board_size - 1), board_size**2))  # Bottom
    for row in range(board_size):
        edge_indices.add(row * board_size)  # Left
        edge_indices.add(row * board_size + (board_size - 1))  # Right
    red_count = sum(1 for idx in edge_indices if board[idx] == 'X')
    blue_count = sum(1 for idx in edge_indices if board[idx] == 'O')
    if red_count > blue_count:
        return 'E'  # Red controls
    else:
        return 'F'  # Blue controls


def calculate_piece_count(board):
    red_count = np.sum(board == 'X')
    blue_count = np.sum(board == 'O')
    if red_count > blue_count:
        return 'G'  # Red controls
    else:
        return 'H'  # Blue controls


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
        return 'I'  # Red controls
    else :
        return 'J'  # Blue controls

def get_neighbors(row, col, board_size=7):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < board_size and 0 <= c < board_size:
            yield r, c

def minimal_path_length_for_player(board_2d, player, board_size=7):
    if player == 'X':
        # Connect left-right
        start_nodes = [(r,0) for r in range(board_size) if board_2d[r,0] == 'X']
        target_cols = board_size - 1
        visited = set(start_nodes)
        frontier = list(start_nodes)
        while frontier:
            new_frontier = []
            for (r,c) in frontier:
                if c == target_cols:
                    return 0  # Found a direct path of player stones
                for nr, nc in get_neighbors(r,c,board_size):
                    if board_2d[nr,nc] == player and (nr,nc) not in visited:
                        visited.add((nr,nc))
                        new_frontier.append((nr,nc))
            frontier = new_frontier
        return board_size * board_size  # No path
    else:
        # player == 'O'
        # Connect top-bottom
        start_nodes = [(0,c) for c in range(board_size) if board_2d[0,c] == 'O']
        target_row = board_size - 1
        visited = set(start_nodes)
        frontier = list(start_nodes)
        while frontier:
            new_frontier = []
            for (r,c) in frontier:
                if r == target_row:
                    return 0
                for nr, nc in get_neighbors(r,c,board_size):
                    if board_2d[nr,nc] == player and (nr,nc) not in visited:
                        visited.add((nr,nc))
                        new_frontier.append((nr,nc))
            frontier = new_frontier
        return board_size * board_size

def edge_to_edge_distance_for_player(board_2d, player, board_size=7):
    # BFS to find shortest path length (in steps).
    # Count number of steps. Just return the minimal steps needed.
    from collections import deque
    if player == 'X':
        # Red: left to right
        start_positions = [(r,0) for r in range(board_size) if board_2d[r,0] in ['X',' ']]
        target_col = board_size - 1
        visited = set(start_positions)
        queue = deque([(pos,0) for pos in start_positions])  # ((r,c), dist)
        while queue:
            (r,c), dist = queue.popleft()
            if c == target_col:
                return dist
            for nr,nc in get_neighbors(r,c,board_size):
                if board_2d[nr,nc] in ['X',' '] and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append(((nr,nc), dist+1))
        return board_size * board_size
    else:
        # Blue: top to bottom
        start_positions = [(0,c) for c in range(board_size) if board_2d[0,c] in ['O',' ']]
        target_row = board_size - 1
        visited = set(start_positions)
        queue = deque([(pos,0) for pos in start_positions])
        while queue:
            (r,c), dist = queue.popleft()
            if r == target_row:
                return dist
            for nr,nc in get_neighbors(r,c,board_size):
                if board_2d[nr,nc] in ['O',' '] and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append(((nr,nc), dist+1))
        return board_size * board_size

def count_bridge_connections(board_2d, board_size=7):
    def creates_win_if_filled(player, er, ec):
        # Temporarily place player stone
        original = board_2d[er,ec]
        board_2d[er,ec] = player
        if player == 'X':
            # Check minimal path for Red again
            path_len = minimal_path_length_for_player(board_2d,'X',board_size)
        else:
            path_len = minimal_path_length_for_player(board_2d,'O',board_size)
        board_2d[er,ec] = original
        return (path_len == 0)
    red_bridge_count = 0
    blue_bridge_count = 0
    empties = [(r,c) for r in range(board_size) for c in range(board_size) if board_2d[r,c] == ' ']

    for (er,ec) in empties:
        if creates_win_if_filled('X', er, ec):
            red_bridge_count += 1
        if creates_win_if_filled('O', er, ec):
            blue_bridge_count += 1

    return red_bridge_count, blue_bridge_count
    
def cluster_strength(board_2d, player, board_size=7):
    # Find connected components of player's stones
    visited = set()
    def dfs(r,c):
        stack = [(r,c)]
        size = 0
        while stack:
            rr,cc = stack.pop()
            if (rr,cc) not in visited and board_2d[rr,cc] == player:
                visited.add((rr,cc))
                size += 1
                for nr,nc in get_neighbors(rr,cc,board_size):
                    if board_2d[nr,nc] == player and (nr,nc) not in visited:
                        stack.append((nr,nc))
        return size
    component_sizes = []
    for r in range(board_size):
        for c in range(board_size):
            if board_2d[r,c] == player and (r,c) not in visited:
                comp_size = dfs(r,c)
                component_sizes.append(comp_size)
    # Cluster strength = sum of squares of component sizes
    return sum([cs*cs for cs in component_sizes])






# --------------------------------
# PRINT ACTIVE FEATURE INFORMATION
# --------------------------------

def discretize_minimal_path_length(red_mpl, blue_mpl):
    # Red wants to minimize minimal path length; if Red < Blue, Red is better
    return 'K' if red_mpl < blue_mpl else 'L'

def discretize_edge_to_edge_distance(red_dist, blue_dist):
    # Red wants to minimize edge-to-edge dist; if Red < Blue, Red is better
    return 'M' if red_dist < blue_dist else 'N'

def discretize_bridge_connections(red_bridges, blue_bridges):
    # More bridges is better; if Red > Blue, Red is better
    return 'P' if red_bridges > blue_bridges else 'Q'

def discretize_cluster_strength(red_cs, blue_cs):
    # Higher cluster strength is better; if Red > Blue, Red is better
    return 'R' if red_cs > blue_cs else 'S'



features = []
for board in X_cat:
    # Dynamically construct feature dictionaries based on FEATURE_SELECTION
    feature_dict = {}
    board_2d = board.reshape((board_size, board_size))
    if FEATURE_SELECTION[1]:  # ENABLE_MID_CTRL
        feature_dict['mid_control'] = calculate_mid_control_categorical(board, board_size=7)
    if FEATURE_SELECTION[2]:  # ENABLE_CORNER_CONTROL
        feature_dict['corner_control'] = calculate_corner_control(board, board_size=7)
    if FEATURE_SELECTION[3]:  # ENABLE_EDGE_PIECE_COUNT
        feature_dict['edge_piece_count'] = calculate_edge_piece_count(board, board_size=7)
    if FEATURE_SELECTION[4]:  # ENABLE_PIECE_COUNT
        feature_dict['piece_count'] = calculate_piece_count(board)
    if FEATURE_SELECTION[5]:  # ENABLE_NEIGHBOR_COUNT
        feature_dict['high_neighbor_count'] = calculate_high_neighbor_count(board, board_size=7)
    if FEATURE_SELECTION[6]:  # ENABLE_MINIMAL_PATH_LENGTH
        red_mpl = minimal_path_length_for_player(board_2d, 'X', board_size=7)
        blue_mpl = minimal_path_length_for_player(board_2d, 'O', board_size=7)
        feature_dict['minimal_path_length'] = discretize_minimal_path_length(red_mpl, blue_mpl)
    if FEATURE_SELECTION[7]:  # ENABLE_EDGE_TO_EDGE_DISTANCE
        red_dist = edge_to_edge_distance_for_player(board_2d,'X',board_size=7)
        blue_dist = edge_to_edge_distance_for_player(board_2d,'O',board_size=7)
        feature_dict['edge_to_edge_distance'] = discretize_edge_to_edge_distance(red_dist, blue_dist)
    if FEATURE_SELECTION[8]:  # ENABLE_BRIDGE_CONNECTIONS
        red_bridges, blue_bridges = count_bridge_connections(board_2d, board_size=7)
        feature_dict['bridge_connections'] = discretize_bridge_connections(red_bridges, blue_bridges)
    if FEATURE_SELECTION[9]:  # ENABLE_CLUSTER_STRENGTH
        red_cs = cluster_strength(board_2d,'X',board_size=7)
        blue_cs = cluster_strength(board_2d,'O',board_size=7)
        feature_dict['cluster_strength'] = discretize_cluster_strength(red_cs, blue_cs)


    features.append(feature_dict)

# Function to dynamically convert feature values to binary
def convert_feature_values(features):
    converted_features = []
    for feature in features:
        converted_feature = {}
        for key, value in feature.items():
            if isinstance(value, str) and len(value) == 1 and value.isalpha():
                # Dynamically compute binary value using ASCII subtraction
                base_letter = 'A' if value in 'AB' else 'C' if value in 'CD' else 'E' if value in 'EF' else 'G' if value in 'GH' else 'I' if value in 'IJ' else 'K' if value in 'KL' else 'M' if value in 'MN' else 'P' if value in 'PQ' else 'R'
                binary_value = ord(value) - ord(base_letter)
                converted_feature[key] = binary_value
            else:
                converted_feature[key] = value
        converted_features.append(converted_feature)
    return converted_features

# Convert features dynamically
converted_features = convert_feature_values(features)

# Print the converted features
print(f"\nExtracted features (filtered and converted):")
print(f"Winners: {y_encoded[:10]}")
for i, feature in enumerate(converted_features[:10]):
    print(f"\n{feature}")





# ------------------------------------
# 4. Splitting, removing duplicates, and balancing datasets
# ------------------------------------


def preprocess_labels(features, labels):

    combined = list(zip(features, labels))  # Preserve exact mapping
    unique_combined = []  # Store unique (feature, label) pairs

    seen = set()
    for feature, label in combined:
        feature_tuple = tuple(feature.flatten())  # Ensure hashability
        if feature_tuple not in seen:
            unique_combined.append((feature, label))
            seen.add(feature_tuple)
    
    unique_features, unique_labels = zip(*unique_combined)

    # Convert back to numpy arrays
    unique_features = np.array(unique_features)
    unique_labels = np.array(unique_labels)

    # Balance the dataset
    label_counts = Counter(unique_labels)
    min_count = min(label_counts.values())
    balanced_indices = []

    for label in label_counts:
        # Get indices of the current label
        indices = np.where(unique_labels == label)[0]

        # Randomly sample indices to match the smallest class count
        sampled_indices = np.random.choice(indices, min_count, replace=False)
        balanced_indices.extend(sampled_indices)

    # Shuffle balanced indices
    np.random.shuffle(balanced_indices)

    return unique_features[balanced_indices], unique_labels[balanced_indices]



def calculate_win_percentage(labels, player_label=0):
    total_games = len(labels)
    player_wins = np.sum(labels == player_label)
    return (player_wins / total_games) * 100


def verify_mapping(orig_X, orig_y, features, processed_X, processed_y):
    # Ensure data sizes match
    assert len(processed_y) <= len(orig_y), \
        f"Processed labels should not exceed original labels. Original: {len(orig_y)}, Processed: {len(processed_y)}"

    for i, (orig, processed) in enumerate(zip(orig_y, processed_y)):
        try:
            print(f"Original Label: {orig}, Processed Label: {processed}")
            print(f"Extracted Features: {features[i]}")
            print(f"Processed Features: {processed_X[i]}")
        except IndexError as e:
            print(f"IndexError at index {i}: {str(e)}")
            print(f"Skipping due to mismatch in data sizes.")
            continue



def calculate_loss(self, true_properties, predicted_properties):
    # Example: Mean squared error on properties
    loss = 0.0
    for graph_id in range(self.number_of_graphs):
        for node_id in range(self.number_of_graph_nodes[graph_id]):
            true_val = true_properties[graph_id][node_id]
            pred_val = predicted_properties[graph_id][node_id]
            loss += (true_val - pred_val) ** 2
    return loss / self.number_of_nodes




X_train, X_temp, y_train, y_temp = train_test_split(X_cat, y_encoded, test_size=val_test_split_ratio, random_state=random_state, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)



print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Calculate win percentages before preprocessing
train_win_percentage = calculate_win_percentage(y_train, player_label=0)
val_win_percentage = calculate_win_percentage(y_val, player_label=0)
test_win_percentage = calculate_win_percentage(y_test, player_label=0)

# Verify class distribution
print(f"\nTraining label distribution (before): {Counter(y_train)}")  
print(f"Validation label distribution (before): {Counter(y_val)}")  
print(f"Testing label distribution (before): {Counter(y_test)}")  

print(f"\nPlayer 1 Win Percentage in Training Set: {train_win_percentage:.2f}%")
print(f"Player 1 Win Percentage in Validation Set: {val_win_percentage:.2f}%")
print(f"Player 1 Win Percentage in Testing Set: {test_win_percentage:.2f}%")


# Preprocess training, validation, and testing datasets
X_train_processed, y_train_processed = preprocess_labels(X_train, y_train)
X_val_processed, y_val_processed = preprocess_labels(X_val, y_val)
X_test_processed, y_test_processed = preprocess_labels(X_test, y_test)


# Calculate win percentages after preprocessing
train_win_percentage = calculate_win_percentage(y_train_processed, player_label=0)
val_win_percentage = calculate_win_percentage(y_val_processed, player_label=0)
test_win_percentage = calculate_win_percentage(y_test_processed, player_label=0)

logging.info(f"\nTraining label distribution (after): {Counter(y_train_processed)}")  # Expected: ~Balanced
logging.info(f"Validation label distribution (after): {Counter(y_val_processed)}")    # Expected: ~Balanced
logging.info(f"Testing label distribution (after): {Counter(y_test_processed)}")      # Expected: ~Balanced

print(f"\nPlayer 1 Win Percentage in Training Set (Processed): {train_win_percentage:.2f}%")
print(f"Player 1 Win Percentage in Validation Set (Processed): {val_win_percentage:.2f}%")
print(f"Player 1 Win Percentage in Testing Set (Processed): {test_win_percentage:.2f}%\n")



# ----------------------------
# 3. Graph Construction
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

adjacency = generate_hex_adjacency(board_size)

# Verify adjacency for the first few nodes
for node_id in range(7):
    print(f"Node {node_id} neighbors: {adjacency[node_id]}")



# Initialize Graphs objects
graphs_train = Graphs(
    number_of_graphs=X_train_processed.shape[0],
    symbols=symbols,  # Should include all possible values in X_cat_train
    hypervector_size=hypervector_size,
    hypervector_bits=hypervector_bits,
    double_hashing=double_hashing
)

graphs_val = Graphs(
    number_of_graphs=X_val_processed.shape[0],
    init_with=graphs_train
)

graphs_test = Graphs(
    number_of_graphs=X_test_processed.shape[0], 
    init_with=graphs_train
)


# ----------------------------
# 4. Encoding Graphs
# ----------------------------

def encode_graphs(graphs, X_data, adjacency, board_size=7):

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

    
    # Step 5: Precompute features
    features = []
    for board in X_data:
        feature_dict = {}
        board_2d = board.reshape((board_size, board_size))
        if FEATURE_SELECTION[1]:  # ENABLE_MID_CTRL
            feature_dict['mid_control'] = calculate_mid_control_categorical(board, board_size)
        if FEATURE_SELECTION[2]:  # ENABLE_CORNER_CONTROL
            feature_dict['corner_control'] = calculate_corner_control(board, board_size)
        if FEATURE_SELECTION[3]:  # ENABLE_EDGE_PIECE_COUNT
            feature_dict['edge_piece_count'] = calculate_edge_piece_count(board, board_size)
        if FEATURE_SELECTION[4]:  # ENABLE_PIECE_COUNT
            feature_dict['piece_count'] = calculate_piece_count(board)
        if FEATURE_SELECTION[5]:  # ENABLE_NEIGHBOR_COUNT
            feature_dict['high_neighbor_count'] = calculate_high_neighbor_count(board, board_size)
        if FEATURE_SELECTION[6]:  # ENABLE_MINIMAL_PATH_LENGTH
            red_mpl = minimal_path_length_for_player(board_2d,'X',board_size)
            blue_mpl = minimal_path_length_for_player(board_2d,'O',board_size)
            feature_dict['minimal_path_length'] = discretize_minimal_path_length(red_mpl, blue_mpl)
        if FEATURE_SELECTION[7]:  # ENABLE_EDGE_TO_EDGE_DISTANCE
            red_dist = edge_to_edge_distance_for_player(board_2d,'X',board_size)
            blue_dist = edge_to_edge_distance_for_player(board_2d,'O',board_size)
            feature_dict['edge_to_edge_distance'] = discretize_edge_to_edge_distance(red_dist, blue_dist)
        if FEATURE_SELECTION[8]:  # ENABLE_BRIDGE_CONNECTIONS
            red_bridges, blue_bridges = count_bridge_connections(board_2d, board_size)
            feature_dict['bridge_connections'] = discretize_bridge_connections(red_bridges, blue_bridges)
        if FEATURE_SELECTION[9]:  # ENABLE_CLUSTER_STRENGTH
            red_cs = cluster_strength(board_2d,'X',board_size)
            blue_cs = cluster_strength(board_2d,'O',board_size)
            feature_dict['cluster_strength'] = discretize_cluster_strength(red_cs, blue_cs)

        features.append(feature_dict)
        
    # Step 6: Add node properties
    precomputed_features = {graph_id: features[graph_id] for graph_id in range(X_data.shape[0])}
    for graph_id in range(X_data.shape[0]):
        for node_id in range(board_size ** 2):
            sym = X_data[graph_id][node_id]
            graphs.add_graph_node_property(graph_id, node_id, sym)
            if ENABLE_CHEAT:
                graphs.add_graph_node_property(graph_id, node_id, str(y_train_processed[graph_id]))
            if ENABLE_MID_CTRL:
                graphs.add_graph_node_property(graph_id, node_id, precomputed_features[graph_id]["mid_control"])
            if ENABLE_CORNER_CONTROL:
                graphs.add_graph_node_property(graph_id, node_id, precomputed_features[graph_id]["corner_control"])
            if ENABLE_EDGE_PIECE_COUNT:
                graphs.add_graph_node_property(graph_id, node_id, precomputed_features[graph_id]["edge_piece_count"])
            if ENABLE_PIECE_COUNT:
                graphs.add_graph_node_property(graph_id, node_id, precomputed_features[graph_id]["piece_count"])
            if ENABLE_NEIGHBOR_COUNT:
                graphs.add_graph_node_property(graph_id, node_id, precomputed_features[graph_id]["high_neighbor_count"])
            if ENABLE_MINIMAL_PATH_LENGTH:
                graphs.add_graph_node_property(graph_id, node_id, precomputed_features[graph_id]["minimal_path_length"])
            if ENABLE_EDGE_TO_EDGE_DISTANCE:
                graphs.add_graph_node_property(graph_id, node_id, precomputed_features[graph_id]["edge_to_edge_distance"])
            if ENABLE_BRIDGE_CONNECTIONS:
                graphs.add_graph_node_property(graph_id, node_id, precomputed_features[graph_id]["bridge_connections"])
            if ENABLE_CLUSTER_STRENGTH:
                graphs.add_graph_node_property(graph_id, node_id, precomputed_features[graph_id]["cluster_strength"])


    # Step 7: Add edges, including Buddy and Not Buddy
    for graph_id in range(X_data.shape[0]):
        for node_id, neighbors in enumerate(adjacency):
            x_count, o_count, empty_count, counter = 0, 0, 0, 0
            for neighbor in neighbors:
                neighbor_value = X_data[graph_id][neighbor]
                if X_data[graph_id][node_id] == neighbor_value and neighbor_value == 'X':
                    edge_type = "BuddyX"
                elif X_data[graph_id][node_id] == neighbor_value and neighbor_value == 'O':
                    edge_type = "BuddyO"
                elif X_data[graph_id][node_id] == neighbor_value and neighbor_value == ' ':
                    edge_type = "Empty Buddy"
                else:
                    edge_type = "Not Buddy"
                graphs.add_graph_node_edge(graph_id, node_id, neighbor, edge_type)
    
            #print(f"Node {node_id} in Graph {graph_id}: X Count = {x_count}, O Count = {o_count}, Empty Count = {empty_count}, edge_type = {edge_type}")

# Encode training and testing graphs
logging.info("Encoding graphs...")
encode_graphs(graphs_train, X_train_processed, adjacency, board_size)
encode_graphs(graphs_val, X_val_processed, adjacency, board_size)
encode_graphs(graphs_test, X_test_processed, adjacency, board_size)



# Finalize encoding
logging.info("Finalizing encoding for graphs...")
graphs_train.encode()
graphs_val.encode()
graphs_test.encode()






# ---------------------------------
# 6. OPTIONAL tuning of s parameter
# ---------------------------------

overall_best_s = s
if ENABLE_TUNING_s == True:
    iteration = 0
    train_accuracies_s = []
    val_accuracies_s = []
    s_values = []
    s_val = s
    overall_best_s = 0
    overall_best_val_accuracy = 0  
    while overall_best_val_accuracy < 1 and iteration < MAX_TRIALS:
        # Reset per iteration
        current_best_val_accuracy = 0  # Best validation accuracy for this value of `s`
        current_best_train_accuracy = 0
        s = s_val + (iteration * Increment_s_pr_trial_by)
        
        # Initialize the MultiClassGraphTsetlinMachine
        tm = MultiClassGraphTsetlinMachine(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            depth=depth,
            message_size=message_size,
            message_bits=message_bits,
            max_included_literals=max_included_literals,
            grid=(16 * 13, 1, 1),  
            block=(128, 1, 1)       
        )
        
        logging.info(f"\nBegin tuning iteration #{iteration + 1}....")
        logging.info(f"s value: {s}")
        start_training = time()
        
        for i in range(EPOCHS_PER_TRIAL):
            trial_time = time()
            # Train the Tsetlin Machine
            tm.fit(graphs_train, y_train_processed, epochs=1, incremental=True)

            # Predict on training data
            train_preds = tm.predict(graphs_train)
            train_accuracy = np.mean(train_preds == y_train_processed)
        
            # Predict on validation data
            val_preds = tm.predict(graphs_val)
            val_accuracy = np.mean(val_preds == y_val_processed)

            # Update current best accuracies for this `s`
            if train_accuracy > current_best_train_accuracy:
                current_best_train_accuracy = train_accuracy
    
            if val_accuracy > current_best_val_accuracy:
                current_best_val_accuracy = val_accuracy
    
            # Update overall best accuracy and best `s` value across all iterations
            if val_accuracy > overall_best_val_accuracy:
                overall_best_val_accuracy = val_accuracy
                overall_best_s = s

            logging.info(f"Trial epoch#{i + 1} -- Accuracy train: {train_accuracy:.2f} -- Accuracy val: {val_accuracy:.2f} -- Duration: {time() - trial_time:.2f} seconds.")
            
        iteration += 1
        train_accuracies_s.append(current_best_train_accuracy)
        val_accuracies_s.append(current_best_val_accuracy)
        s_values.append(s)
    
        logging.info(f"T: {T}, s: {s}, Best val acc: {current_best_val_accuracy:.3f}.")
        logging.info(f"Best s value trial run: {overall_best_s} at iteration {(s-1)/5:.3f}.")
        logging.info(f"Total time: {time() - start_training:.2f} seconds.\n")
        
    logging.info(f"\nOverall best s value: {overall_best_s} with val accuracy: {overall_best_val_accuracy:.3f}")

    # Plot Training and Validation Accuracies vs s Values
    plt.figure(figsize=(14, 8))  # Set figure size
    plt.plot(s_values, train_accuracies_s, marker='o', linestyle='-', label='Training Accuracy', color='blue')
    plt.plot(s_values, val_accuracies_s, marker='o', linestyle='--', label='Validation Accuracy', color='purple')
    plt.title('Training and Validation Accuracies vs s Values')
    plt.xlabel('s values')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')  # Automatically position the legend
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to a file and display it
    filename = f'Accuracies_vs_s_BestValAcc-{overall_best_val_accuracy}_s-{overall_best_s}.png'
    plt.savefig(filename)
    logging.info(f"Plot saved as {filename}")
    plt.show()
            





# -------------------------------------
# 6. Training the Graph Tsetlin Machine
# -------------------------------------


tm = MultiClassGraphTsetlinMachine(
    number_of_clauses=number_of_clauses,
    T=T,
    s=overall_best_s,
    depth=depth,
    message_size=message_size,
    message_bits=message_bits,
    max_included_literals=max_included_literals,
    grid=(16 * 13, 1, 1),   # Adjust based on system capabilities
    block=(128, 1, 1)       # Adjust based on system capabilities
)
    
# Initialize lists to store metrics
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

logging.info("Starting training...")
start_training = time()
for epoch in range(epochs):
    print("")
    logging.info(f"\tEpoch {epoch + 1}/{epochs}: ")
    
    # Record training start time
    epoch_start = time()
    
    # Train for one epoch
    tm.fit(graphs_train, y_train_processed, epochs=1, incremental=True)

    # Predict on training data
    train_preds = tm.predict(graphs_train)
    train_accuracy = np.mean(train_preds == y_train_processed)
    train_accuracies.append(train_accuracy)
    
    # Predict on validation data
    val_preds = tm.predict(graphs_val)
    val_accuracy = np.mean(val_preds == y_val_processed)
    val_accuracies.append(val_accuracy)

    # Calculate loss 
    train_loss = np.mean((np.array(y_train_processed) - np.array(train_preds)) ** 2)
    val_loss = np.mean((np.array(y_val_processed) - np.array(val_preds)) ** 2)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    training_time = time() - epoch_start
    logging.info(f"\tTrain Accuracy: \t\t{train_accuracy * 100:.2f}%")
    logging.info(f"\tTrain Loss: \t\t\t{train_loss:.3f}")
    logging.info(f"\tValidation Accuracy: \t\t{val_accuracy * 100:.2f}%")
    logging.info(f"\tValidation Loss: \t\t{val_loss:.3f}")
    logging.info(f"\tSample Train Predictions:\t{train_preds[:20]}, Labels: {y_train_processed[:20]}")
    logging.info(f"\tSample Val Predictions:\t\t{val_preds[:20]}, Labels: {y_val_processed[:20]}")
    logging.info(f"\tTraining Time:\t\t\t{training_time:.2f} seconds.")
    
# Predict on testing data
test_preds = tm.predict(graphs_test)
test_accuracy = np.mean(test_preds == y_test_processed)

total_training_time = time() - start_training
logging.info(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
logging.info(f"Total Training Time: {total_training_time:.2f} seconds.")


# Plotting Accuracies Over Epochs
plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), [acc * 100 for acc in train_accuracies], label='Train Accuracy')
plt.plot(range(1, epochs + 1), [acc * 100 for acc in val_accuracies], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title(f'Training and Testing Accuracy Over Epochs, Version: {VERSION}')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the accuracy plot
accuracy_plot_filename = f'overall_accuracy_Epochs-{epochs}_Clauses-{number_of_clauses}_Features-{FEATURE_SELECTION}_TestAccuracy-{test_accuracy * 100:.2f}_Start-{timestamp}.png'
plt.savefig(accuracy_plot_filename)
logging.info(f"Accuracy plot saved as {accuracy_plot_filename}")
plt.close()


plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the loss plot
loss_plot_filename = f'loss_plot_Epochs-{epochs}_Clauses-{number_of_clauses}_Features-{FEATURE_SELECTION}_Start-{timestamp}.png'
plt.savefig(loss_plot_filename)
logging.info(f"Loss plot saved as {loss_plot_filename}")
plt.show()




# ----------------------------
# 7. Evaluating and Interpreting the Model
# ----------------------------

# Retrieve model clauses
weights = tm.get_state()[1].reshape(2, -1)  # Assuming two classes

if test_accuracy > 70:
    print("\n--- Clauses ---\n")
    for clause_idx in range(tm.number_of_clauses):
        literals = []
        for hv_idx in range(hypervector_size * 2):
            if tm.ta_action(0, clause_idx, hv_idx):
                if hv_idx < hypervector_size:
                    literals.append(f"x{hv_idx}")               # Positive literal
                else:
                    literals.append(f"NOT x{hv_idx - hypervector_size}")  # Negative literal
        clause_expression = " AND ".join(literals)
        if len(literals) != 0: 
            logging.info(f"Clause #{clause_idx + 1}:")
            logging.info(f"  Weights: (Red: {weights[0, clause_idx]}, Blue: {weights[1, clause_idx]})")
            logging.info(f"  Expression: {clause_expression}")
            logging.info(f"  Number of Literals: {len(literals)}\n")

# Classification Report and Confusion Matrix
logging.info("Training Classification Report:")
logging.info("\n" + classification_report(y_train_processed, train_preds))

logging.info("Validation Classification Report:")
logging.info("\n" + classification_report(y_val_processed, val_preds))

logging.info("Testing Classification Report:")
logging.info("\n" + classification_report(y_test_processed, test_preds))

logging.info("Confusion Matrix (Test Set):")
logging.info(confusion_matrix(y_test_processed, test_preds))
