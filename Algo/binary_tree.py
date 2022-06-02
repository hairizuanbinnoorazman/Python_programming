# Adding items to binary tree
class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val

node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)
node5 = Node(5)

node1.left = node2
node1.right = node3
node2.left = node4
node2.right = node5

# Traverse it in order
def traverse_in_order(root):
    left_items = []
    right_items = []
    if root.left is not None:
        left_items = traverse_in_order(root.left)
    if root.right is not None:
        right_items = traverse_in_order(root.right)
    combined = left_items + [root.val] + right_items
    return combined

print("traverse in order")
items_in_order = traverse_in_order(node1)
print(items_in_order)

# Traverse it pre order
def traverse_pre_order(root):
    left_items = []
    right_items = []
    if root.left is not None:
        left_items = traverse_pre_order(root.left)
    if root.right is not None:
        right_items = traverse_pre_order(root.right)
    combined = [root.val] + left_items + right_items
    return combined

print("traverse pre order")
items_in_order = traverse_pre_order(node1)
print(items_in_order)


# Traverse it post order
def traverse_post_order(root):
    left_items = []
    right_items = []
    if root.left is not None:
        left_items = traverse_post_order(root.left)
    if root.right is not None:
        right_items = traverse_post_order(root.right)
    combined = left_items + right_items + [root.val]
    return combined

print("traverse post order")
items_in_order = traverse_post_order(node1)
print(items_in_order)
