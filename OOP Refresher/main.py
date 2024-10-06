import random

class Die:

    def __init__(self, num_sides=6):
        if isinstance(num_sides, int) and num_sides > 1:
            self._num_sides = num_sides
            self._value = None
        else:
            print("please enter a valid number of sides")

    @property
    def value(self):
        return self._value 

    def roll(self):
        new_value = random.randint(1,self._num_sides)
        self._value = new_value
        return new_value
    
class Player:

    def __init__(self, die, is_computer=False):
        self._counter = 10
        self._die = die
        self._is_computer = is_computer

    @property
    def die(self):
        return self.die

    @property
    def is_computer(self):
        return self._is_computer

    @property
    def counter(self):
        return self.counter
    
    def decrement_counter(self):
        self.counter -= 1

    def increment_counter(self):
        self.counter += 1

    def roll_die(self):
        return self._die.roll()
    
my_die = Die()
my_player = Player(my_die, True)
print(my_player)
print(my_player.is_computer)