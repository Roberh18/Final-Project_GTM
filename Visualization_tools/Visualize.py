import pandas as pd

CSV_FILE = "hex_games_100000_size_7_BeforeEnd-0_OpenPos-18_Random-True.csv"
BOARD_DIM = 7
GAMES_TO_SHOW = 5  # How many games from the top to visualize

def print_hex_board(board, board_dim=BOARD_DIM):
    """
    Print a hex board state from a 2D array where:
    1  -> 'X' (Red)
    -1 -> 'O' (Blue)
    0  -> '.' (Empty)
    """
    symbol_map = {1: 'X', -1: 'O', 0: '.'}
    
    for i in range(board_dim):
        print(" " * i, end="")
        for j in range(board_dim):
            val = board[i][j]
            symbol = symbol_map[val]
            print(f" {symbol}", end="")
        print() 

# Load the dataset
data = pd.read_csv(CSV_FILE)

# Verify structure
columns = [f'cell{i}_{j}' for i in range(BOARD_DIM) for j in range(BOARD_DIM)] + ['winner']
if list(data.columns) != columns:
    print("Warning: The columns in the CSV do not match the expected structure.")

# Pick the first N games to visualize
subset = data.head(GAMES_TO_SHOW)

for idx, row in subset.iterrows():
    board_array = []
    for i in range(BOARD_DIM):
        row_cells = []
        for j in range(BOARD_DIM):
            cell_value = int(row[f'cell{i}_{j}'])  # Convert '1', '-1', '0' strings to int
            row_cells.append(cell_value)
        board_array.append(row_cells)

    winner = int(row['winner'])  # '1' for Red, '-1' for Blue
    winner_symbol = "Red (X)" if winner == 1 else "Blue (O)"

    print(f"\nGame #{idx+1} (Row {idx}) - Winner: {winner_symbol}")
    print_hex_board(board_array, BOARD_DIM)



'''
hex_games_20000_size_5_BeforeEnd-0_OpenPos-9_Random-True.csv
hex_games_20000_size_5_BeforeEnd-3_OpenPos-14_Random-True.csv
hex_games_20000_size_5_BeforeEnd-5_OpenPos-18_Random-True.csv
'''