class Character:

    def __init__(self, x, y, num_lives):
        self.x = x
        self.y = y
        self.num_lives = num_lives


class Player(Character):
    INITIAL_X = 0
    INITIAL_Y = 0
    INITIAL_NUM_LIVES = 10

    def __init__(self, score=0):
        Character.__init__(self, Player.INITIAL_X, Player.INITIAL_Y, Player.INITIAL_NUM_LIVES)


class Enemy(Character):

    def __init__(self, x=15, y=15, num_lives=8, is_poisonous=False):
        Character.__init__(self, x, y, num_lives)
        self.is_poisonous = is_poisonous