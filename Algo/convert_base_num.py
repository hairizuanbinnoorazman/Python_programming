# Convert the number to a different base
# e.g. base10 to base5
# Convert 7 to base5
# 1 2 3 4 10 11 12 13 14 20
#
# Convert 10 to base 7
# 1 2 3 4 5 6 10 11 12 13
#
# Convert 15 to base 3
# 1 2 10 11 12 20 21 22 100 101 102 110 111 112 120
#
# Convert 5 to base 2
# 1 10 11 100 101

def convert_base(convert_to_base: int, num: int):
    is_negative = False
    temp_num = num
    if num < 0:
        is_negative = True
        temp_num = -1 * temp_num

    multiplier = 1
    final = 0
    while temp_num > 0:
        remainder = temp_num % convert_to_base
        final = final + remainder * multiplier
        temp_num = temp_num - remainder
        temp_num = temp_num / convert_to_base
        multiplier = multiplier * 10

    if is_negative:
        final = final * -1

    return final

print(convert_base(5, 7))
print(convert_base(3, 14))
print(convert_base(2, 5))