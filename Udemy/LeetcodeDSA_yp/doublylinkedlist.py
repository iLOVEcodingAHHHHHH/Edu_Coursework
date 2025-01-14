class Node:
    def __init__(self, value):
        self.next = None
        self.prev = None
        self.value = value

class DblList:
    def __init__(self, value):
        self.head = self.tail = Node(value)
        self.length = 1

    def append(self, value):
        new_node = Node(value)
        if self.length == 0:
            self.head = self.tail = new_node
            return True
        self.tail.next = new_node
        new_node.prev = self.tail
        self.tail = new_node
        self.length += 1

    def prepend(self, value):
        new_node = Node(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.length += 1

        
    def pop_last(self):
        if not self.head:
            return None
        temp = self.tail
        if self.length == 1:
            self.head = None
        else:
            self.tail = temp.prev
            self.tail.next = None
            temp.prev = None
        self.length -= 1

        return temp
    
    def pop_first(self):
        if not self.head:
            return None
        temp = self.head
        if self.length == 1:
            self.head = self.tail = None
        else:
            self.head = temp.next
            self.head.prev = None
            temp.next = None
        self.length -= 1
        return temp

    def prepend(self, value):
        new_node = Node(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.length += 1
        return True



    def get(self, index):
        
        if not 0 <= index <self.length:
            return None
        if index < self.length/2:
            temp = self.head
            for _ in range(index):
                temp = temp.next
        else:
            temp = self.tail
            for _ in range(self.length -1, index, -1):
                temp = temp.prev
        return temp

    def set_value(self, index, value):
        temp = self.get(index)
        if temp:
            temp.value = value
            return True
        return False

    def insert(self, index, value):
        if not 0 <= index <= self.length:
            return None
        if index == self.head:
            return self.prepend(value)
        if index == self.length:
            return self.append(value)
        new_node = Node(value)
        after = self.get(index)
        before = after.prev
        new_node.prev = before
        new_node.next = after
        before.next = after.prev = new_node
        self.length += 1
        return True
    
    def remove(self, index):
        if not 0 <= index < self.length:
            return None
        if index == 0:
            return self.pop_first()
        if index == self.length - 1:
            return self.pop_last()
        temp = self.get(index)
        temp.prev.next = temp.next
        temp.next.prev = temp.prev
        temp.prev = temp.next = None
        self.length -= 1
        return temp