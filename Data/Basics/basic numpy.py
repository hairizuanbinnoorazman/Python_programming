import numpy as np

# Declaring simple array
a = np.array([1, 2, 3])

# Declaring multiple array
b = np.array([[1, 2], [3, 4]])

# Python generator
# Use next to get the next value
def firstn(n):
    num = 0
    while num < n:
        yield [num, num+1, num+2]
        num = num + 1