

def display_board(observation):
   
    """
    Display the board to the terminal
    """
    p1_plane = observation["observation"][:, :, 0]
    p2_plane = observation["observation"][:, :, 1]

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
    
    
   

    

    