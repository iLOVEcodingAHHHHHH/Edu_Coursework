def r_contains(self, value):
    return self.__r_contains(self.root, value)

def __r_contains(self, current_node, value):
    if current_node == None:
        return False
    if value == current_node.value:
        return True
    if value < current_node.value:
        return self__r_contains(current_node.left, value)
    if value > current_node.value:
        return self__r_contains(current_node.right, value)

def r_insert(self, value):
    self.__r_insert(self.root, value)

def __r_insert(self, current_node, value):
    if current_node == None:
        return Node(value)
    if value < current_node.value:
        current_node.left = self.__r_insert(current_node, value)
    if value > current_node.value:
        current_node.right = self.__r_insert(current_node.right, value)
    return current_node


def min_value(self, current_node):
    while current_node.left is not None:
        current_node = current_node.left
    return current_node.value


def __delete_node(self, current_node, value):
    if current_node == None:
        return None
    if value < current_node.value:
        current_node.left = self.__delete_node(current_node.left, value)
    elif value > current_node.value:
        current_node.right = self.__delete_node(current_node.right, value)
    
    else:
        if not any(current_node.left, current_node.right):
            return None
        elif current_node.left == None:
            current_node = current_node.right
        elif current_node.right == None:
            current_node = current_node.left
        else:
            sub_tree_min_value = self.min_value(current_node.right)
            current_node.value = sub_tree_min_value
            current_node.right = self.__delete_node(current_node.right, sub_tree_min_value)
    return current_node

def delete_node(self, value):
    self.__delete_node(self.root, value)