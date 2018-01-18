
import pandas as pd

import hashlib


s = "Hello Word"


print(hash(s))


print(hashlib.algorithms_available)


hash_object = hashlib.sha1(b'Hello World')

hex_dig = hash_object.hexdigest()


def check_collision(l, hash_max):
    myset = set(l)

    print("Input#" + str(len(l)) + " uniques# " + str(len(myset)))

    l = list(myset)

    hash_list = []

    for x in l:
#        hash_object = hashlib.sha512(x.encode())

        uhash = hash(x) % hash_max
        hash_list.append(uhash)

#        hex_dig = hash_object.hexdigest()
#        hash_list.append(hex_dig)


    hash_set = set(hash_list)

    nElements = len(l)
    nHash     = len(hash_set)

    print("Elements# " + str(nElements) + " hash uniques# " + str(nHash))

w = 90

check_collision(l, 900000)

# Elements#162032 hash uniques# 162029
check_collision(l, 2**32 -1)

check_collision(l, 2**29 -1)

check_collision(l, 2**64 -1)