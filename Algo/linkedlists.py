class Node:
    def __init__(self, num):
        self.num = num
        self.next = None


node1 = Node(12)
node2 = Node(13)
node3 = Node(14)
node4 = Node(15)
node1.next = node2
node2.next = node3
node3.next = node4

iterNode = node1

# Printing
print("Begin printing list")
while True:
   print(iterNode.num)
   iterNode = iterNode.next
   if iterNode == None:
       break
print("End printing list")

# Searching
def search(iterNode, num):
    while True:
        if iterNode.num == num:
            print("found {}".format(num))
            break
        iterNode = iterNode.next
        if iterNode == None:
            print("Not found")
            break

search(node1, 1)
search(node1, 14)