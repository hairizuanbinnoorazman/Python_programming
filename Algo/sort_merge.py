# Merge sort
def merge_sort(list_items):
    # print(list_items)
    if len(list_items) == 1:
        return list_items
    left_side = list_items[0:int(len(list_items)/2)]
    right_side = list_items[int(len(list_items)/2):len(list_items)]
    arranged = []
    left_counter = 0
    right_counter = 0
    sorted_left_side = merge_sort(left_side)
    sorted_right_side = merge_sort(right_side)

    while left_counter < len(sorted_left_side) or right_counter < len(sorted_right_side):
        # print("{l} {r}".format(l=sorted_left_side,r=sorted_right_side))
        if left_counter >= len(sorted_left_side):
            # print("first")
            arranged = arranged + sorted_right_side[right_counter:len(sorted_right_side)]
            right_counter = len(sorted_right_side)
        elif right_counter >= len(sorted_right_side):
            # print("second")
            arranged = arranged + sorted_left_side[left_counter:len(sorted_left_side)]
            left_counter = len(sorted_left_side)
        elif sorted_left_side[left_counter] <=  sorted_right_side[right_counter]:
            # print("third")
            arranged = arranged + [sorted_left_side[left_counter]]
            left_counter = left_counter + 1
        else:
            # print("fourth")
            arranged = arranged + [sorted_right_side[right_counter]]
            right_counter = right_counter + 1
    return arranged

items = [13, 12, 11, 10, 9, 5, 3, 2, 1]
sorted = merge_sort(items)
print("output: {s}".format(s=sorted))