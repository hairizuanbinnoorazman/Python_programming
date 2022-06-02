# Via Node construct
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

node1 = Node(1)
node2 = Node(1)
node3 = Node(2)
node4 = Node(3)
node5 = Node(3)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5

head_node = node1
def printer(head):
    while True:
        if head == None:
            break
        print(head.val)
        head = head.next

printer(head_node)
print("stop initial printing of values")

print("removing duplicates")
current_node = node1.next
previous_node = node1
previous_val = node1.val
while True:
    if current_node == None:
        previous_node.next = None
        break
    elif current_node.val == previous_val:
        current_node = current_node.next
        continue
    else:
        previous_node.next = current_node
        previous_val = current_node.val
        previous_node = current_node
        current_node = current_node.next
        # printer(node1)
        continue
lol = node1
while True:
    if lol == None:
        break
    print(lol.val)
    lol = lol.next

