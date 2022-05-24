# insertion sort
def insertion_sort(list_items):
    counter = 0
    while counter < len(list_items):
        int_insertion_sort(list_items, counter)
        counter = counter + 1
    return list_items

def int_insertion_sort(list_items, counter):
    print(counter)
    while counter >= 1:
        if list_items[counter] < list_items[counter - 1]:
            temp_store = list_items[counter - 1]
            list_items[counter - 1] = list_items[counter]
            list_items[counter] = temp_store
            print(list_items)
        counter = counter - 1


items = [13, 12, 11, 10, 9, 5, 3, 2, 1]
sorted = insertion_sort(items)
print(sorted)