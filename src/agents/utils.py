import numpy as np

def display_board(observation):
   
    """
    Display the board to the terminal
    """
    current_player_plane = observation["observation"][:, :, 0]
    opponent_plane = observation["observation"][:, :, 1]
    total_pieces = np.sum(current_player_plane) + np.sum(opponent_plane)

    # If total pieces is even, it's player 1's turn (current player is p1)
    # If total pieces is odd, it's player 2's turn (current player is p2)
    if total_pieces % 2 == 0:
        p1_plane, p2_plane = current_player_plane, opponent_plane
    else:
        p2_plane, p1_plane = current_player_plane, opponent_plane

    board = [
            [" "," "," "],
            [" "," "," "],
            [" "," "," "]]
    
    for i in range(3):
        for j in range(3):
            if p1_plane[i,j] == 1:
                board[j][i] = "X"
            if p2_plane[i,j] == 1:
                board[j][i] = "O"

    # print(p1_plane)
    # print(p2_plane)

    print()
    print("BOARD")
    print("=====")
    for i,row in enumerate(board):
        row_disp = ("|").join(row)
        print(row_disp)
        if i < 2:
            print("-----")
    print("=====")
    print()
    
    
   

    

    