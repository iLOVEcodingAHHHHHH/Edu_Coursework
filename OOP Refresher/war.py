class Suit:

    def __init__(self, symbol, description):
        self._symbol = symbol
        self._description = description

    @property
    def symbol(self):
        return self._symbol

    @property
    def description(self):
        return self._description

class Card:

    def __init__(self, suit, value):
        self._suit = suit
        self._value = value
        

    @property
    def suit(self):
        return self._suit
    
    @property
    def value(self):
        return self._value