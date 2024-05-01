import time

print("this is from another.py", flush=True)

try:
    with open("hoho.txt", 'w+') as f:
        f.write("akcmkaca")
        f.flush()
except Exception as e:
    print(e)

# while True:
#     print("start sleeping", flush=True)
#     time.sleep(1)