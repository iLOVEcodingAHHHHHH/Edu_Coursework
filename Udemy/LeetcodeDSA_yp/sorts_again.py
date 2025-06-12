def bubble_sort(my_list):
    for i in range(len(my_list)-1, 0, -1):
        for j in range(i):
            if my_list[j] > my_list[j+1]:
                my_list[j], my_list[j+1] = my_list[j+1], my_list[j]


###
The solution code doesn't currently implement an exit mechanism to stop once the list is sorted.  GPT suggested using break if it reaches the end of the nested loop and no swaps occur, currently it doesn't track whether or not any swapping is happening.  On a side note, here's my alt code taken straight from the lecture:

    def bubble_sort(self):

        for i in range(self.length-1, 0, -1):

            current = self.head

            swapped = False

            for j in range(i):

                if current.value > current.next.value:

                    current.value, current.next.value = current.next.value, current.value

                    swapped = True

                current = current.next

            if not swapped:

                break
###