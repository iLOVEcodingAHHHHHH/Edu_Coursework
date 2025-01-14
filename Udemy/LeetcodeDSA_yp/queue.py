class Node:
    def __init__(self, value):
        self.next = None
        self.value = value

class Queue:
    def __init__(self, value):
        new_node = Node(value)
        self.first = self.last = new_node
        self.length = 1

    def print_queue(self):
        if not self.first:
            print('None')
            return None
        temp = self.first
        while temp:
            print(temp.value)
            temp = temp.next

    def enque(self, value):
        new_node = Node(value)
        if not self.first:
            self.first = self.last = new_node
        else:
            self.last.next = new_node
            self.last = new_node
        self.length += 1
        return True
    
    def dequeue(self):
        if self.length == 0:
            return None
        temp = self.first
        if self.length == 1:
            self.first = self.last = None
        else:
            self.first = temp.next
            temp.next = None
        self.length -= 1
        return temp
        