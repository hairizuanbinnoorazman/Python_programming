# num:    1 2 3 4 5 6
# output: 1 1 2 3 5 8
def fibonacci(num):
    output = []
    initial = 1
    while initial <= num:
        if initial <= 2:
            output = output + [1]
        elif initial > 2:
            output = output + [(output[initial-2] + output[initial-3])]
        initial = initial + 1
    return output

output = fibonacci(6)
print(output)