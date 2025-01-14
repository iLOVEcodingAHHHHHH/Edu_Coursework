class Node:
    def __init__(self, value):
        self.next = None
        self.value = value

class Stack:
    def __init__(self, value):
        self.top = Node(value)
        self.height = 1

    def print(self):
        temp = self.top
        while temp:
            print(temp.value)
            temp = temp.next

    def push(self, value):
        new_node = Node(value)
        new_node.next = self.top
        self.top = new_node
        self.height += 1
        return True
    
    def pop(self):
        if not self.top:
            return None
        temp = self.top
        self.top = temp.next
        temp.next = None
        self.height -= 1
        return temp
