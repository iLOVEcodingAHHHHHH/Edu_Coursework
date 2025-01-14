class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        new_node = Node(value)
        
        if not self.root:
            self.root = new_node
            return True
        
        temp = self.root

        while value != temp.value:
            if value < temp.value:
                if not temp.left:
                    temp.left = new_node
                    return True
                temp = temp.left
            else:
                if not temp.right:
                    temp.right = new_node
                    return True
                temp = temp.right
        return False
        


my_tree = BinarySearchTree()
my_tree.insert(2)
my_tree.insert(1)
my_tree.insert(3)
my_tree.insert(3)

print(my_tree.root.right.value)