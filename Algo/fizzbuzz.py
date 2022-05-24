# fizzbuzz
# For multiple of 3 - fizz
# For multiple of 5 - buzz
# For multiple of 3 and 5 - fizzbuzz
# else - print the number
def fizzbuzz(num):
    iter = 1
    while iter <= num:
        if iter % 3 == 0 and iter % 5 == 0:
            print("fizzbuzz")
        elif iter % 3 == 0:
            print("fizz")
        elif iter % 5 == 0:
            print("buzz")
        else:
            print(iter)
        iter = iter + 1

fizzbuzz(20)