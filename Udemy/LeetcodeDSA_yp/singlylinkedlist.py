class Node:

    def __init__(self, value):
        self.next = None
        self.value = value



class SLL:

    def __init__(self):
        self.head = None
        self.tail = None
        self.len = 0


    def add_to_empty(self, val):
        self.head = self.tail = Node(val)


    def empty_list_chk(self):
        if self.len == 0:
            raise Exception("List is empty")


    def append(self, val):
        new_node = Node(val)
        if self.len == 0:
            self.add_to_empty(val)
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.len += 1
        return True


    def prepend(self, val):
        new_node = Node(val)
        if self.len == 0:
            self.add_to_empty(val)
        else:
            new_node.next = self.head
            self.head = new_node
        self.len += 1
        return True


    def print_values(self):
        self.empty_list_chk()

        point = self.head
        while point is not None:
            print(point.value)
            point = point.next


    def pop_tail(self):
        self.empty_list_chk()

        point = self.head
        follow = self.head
        while point.next:
            follow = point
            point = point.next
        self.tail = follow
        self.tail.next = None
        self.len -= 1
        if self.len == 0:
            self.head = self.tail = None
        return point


    def pop_head(self):
        self.empty_list_chk()

        pop_node = self.head
        self.head = self.head.next
        pop_node.next = None
        if self.len == 1:
            self.tail = None
        self.len -= 1
        
        return pop_node


    def index_bndry_chk(self, index):
        if not (0 <= index < self.len):
            raise Exception("Index not in range.")


    def get_node(self, index):
        self.empty_list_chk()
        self.index_bndry_chk(index)

        point = self.head
        for _ in range(index):
            point = point.next
        return point


    def set_node(self, index, val):

        point = self.get(index)
        if point:
            point.value = val
            return True
        return False


    def insert_node(self, index, val):
        self.index_bndry_chk()
        if index == 0:
            self.prepend(val)
        if index == self.len:
            self.append(val)

        new_node = Node(val)
        point = self.get_node(index - 1)
        new_node.next = point.next
        point.next = new_node
        self.len += 1
        return True


    def remove_node(self, index):
        self.index_bndry_chk()
        if index == 0:
            self.pop_head
        if index == self.len:
            self.pop_tail

        prev_point = self.get_node(index - 1)
        temp = prev_point.next
        prev_point.next =   temp.next
        temp.next = None
        self.len -= 1
        return temp

    def reverse(self):
        if 0 <= self.len <= 1:
            return self

        temp = self.head
        before = None
        for _ in range(self.len):
            after = temp.next
            temp.next = before
            before = temp
            temp = after
            

        self.head, self.tail = self.tail, self.head
        return self