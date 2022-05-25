# Quick sort
def quick_sort(list_items):
    if len(list_items) <= 1:
        return list_items

    pivot_number = list_items[len(list_items)-1]
    left_side = []
    right_side = []
    for n in list_items[0:len(list_items)-1]:
        if n <= pivot_number:
            left_side = left_side + [n]
        else:
            right_side = right_side + [n]

    sorted_left_side = quick_sort(left_side)
    sorted_right_side = quick_sort(right_side)
    arranged = sorted_left_side + [pivot_number] + sorted_right_side
    return arranged

items = [13, 12, 11, 10, 9, 5, 3, 2, 1]
sorted = quick_sort(items)
print(sorted)