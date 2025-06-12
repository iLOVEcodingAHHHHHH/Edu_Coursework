def swap(my_list, i1, i2):
    my_list[i1], my_list[i2] = my_list[i2], my_list[i1]

def pivot(my_list, pivot_index, end_index):
    
    swap_index = pivot_index
    
    for i in range(pivot_index+1, end_index+1):
        if my_list[i] < my_list[pivot_index]:
            swap_index += 1
            swap(my_list, swap_index, i)
    swap(my_list, swap_index, pivot_index)
    return (swap_index)

def quick_sort_helper(my_list, l, r):
    if l < r:
        pivot_index = pivot(my_list, l, r)
        quick_sort_helper(my_list, l, pivot_index-1)
        quick_sort_helper(my_list, pivot_index+1, r)

def quick_sort(my_list):
    return quick_sort_helper(my_list, 0, len(my_list)-1)


my_list = [4,6,1,7,3,2,5]
print(quick_sort(my_list))
print(my_list)