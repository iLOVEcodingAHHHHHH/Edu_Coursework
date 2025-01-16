class HashTable:
    def __init__(self, size = 7):
        self.data_map = [None] * size
    
    def __hash(self, key):
        my_hash = 0
        for letter in key:
            my_hash = (my_hash + ord(letter) * 23) % len(self.data_map)
        return my_hash
    
    def set_item(self, key, value):
        index = self.__hash(key)
        if not self.data_map[index]:
            self.data_map[index] = []
        self.data_map[index].append([key, value])

    def get_item(self, key):
        index = self.__hash(key)
        if self.data_map[index]:
            for _ in self.data_map[index]:
                if _[0] == key:
                    return _[1]
        return None
    
    def keys(self):
        all_keys = []
        for i in self.data_map:
            if i is not None:
                for j in i:
                    all_keys.append(j[0])
        return all_keys

my_ht = HashTable()

my_ht.set_item('bolts', 1400)
my_ht.set_item('washers', 50)
print(my_ht.get_item('bolts'))
print(my_ht.get_item('washers'))
print(my_ht.get_item('lumber'))
print(my_ht.keys())
