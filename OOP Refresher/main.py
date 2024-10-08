import random

class Move:

    def __init__(self,value):
        self._value = value

    @property
    def value(self):
        return self._value

    def is_valid(self):
        return 1 <= self._value <= 9

    def get_row(self):
        return (self._value // 3)

    def get_column(self):
        return (self._value % 3) - 1
    
class Player():

    PLAYER_MARKER = "X"
    COMPUTER_MARKER = "O"

    def __init__(self, is_human=True):
        self._is_human = is_human

        if is_human:
            self._marker = Player.PLAYER_MARKER
        else:
            self._marker = Player.COMPUTER_MARKER

    @property
    def is_human(self):
        return self._is_human

    def marker(self):
        return self._marker

    def get_move(self):
        if self._is_human:
            return self.get_human_move()
        else:
            return self.get_computer_move()

    def get_human_move(self):
        while True:
            user_input = int(input("Please enter your move: "))
            move = Move(user_input)
            if move.is_valid():
                break
            else:
                print("Please enter a integer between 1 and 9")
        return move

    def get_computer_move(self):
        random_choice = random.choice(list(range(1,10)))
        move = Move(random_choice)
        print("Computer move: ", move.value)
        return move

class Board:

    EMPTY_CELL = 0
    
    def __init__(self):
        self.game_board = [[0,0,0],[0,0,0],[0,0,0]]

    def print_board(self):
        print("\nPositions:")
        self.print_board_with_positions()

        print("Board")
        for row in self.game_board:
            print("|",end="")
            for column in row:
                if column == Board.EMPTY_CELL:
                    print("   |", end="")
                else:
                    print(f" {column} |", end="")
            print()
        print()

    def print_board_with_positions(self):
        print("| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n| 7 | 8 | 9 |")

    def submit_move(self, player, move):
        row = move.get_row()
        col = move.get_column()
        value = self.game_board[row][col]

        if value == Board.EMPTY_CELL:
            self.game_board[row][col] = player.marker
        else:
            print("This position is already taken, YOU LOSE A TURN!!!")

board = Board()
player = Player()
move = Move(5)

board.print_board()
board.submit_move(player, move)
board.print_board()