# Sort list
def bubble_sort(list_items):
    while int_bubble_sort(list_items):
        print("Sorting {items}".format(items=list_items))
    return list_items

def int_bubble_sort(list_items):
    counter = 0
    sort_happened = False
    while counter < len(list_items)-1:
        if list_items[counter] > list_items[counter + 1]:
            temp_store = list_items[counter]
            list_items[counter] = list_items[counter + 1]
            list_items[counter + 1] = temp_store
            sort_happened = True
        counter = counter + 1
    return sort_happened


items = [13, 12, 11, 10, 9, 5, 3, 2, 1]
sorted = bubble_sort(items)
print(sorted)
