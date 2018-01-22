
import pandas as pd

import hashlib


# Data value intrinsically/ in themselves identifying the person. 

# => MUST BE XXX


# Data value (combination) able to identify singular person by its specific content.

# => FOR DATA PRIVACY INVESTIGATION


# MUST ALWAYS BE SEEN IN CONTEXT


# Data value (combination) this is not able to disclose singular person.





# SSN number


# Requirements on data

#    Reverse look up SSN based on given information (with, likely, a supplied identifier).

#    (Behind the scenes) combine personal data from several sources.

#    Varying: One-off/rare data combinations from several sources. And/or: Run-time updates with recent/immediate data.



# Attacks

# Figure out who is behind a record of information. From one table. From all tables. (Retrieve the SSN and name from one record).

# Retrieve SSNs at large.

def hash_function(s, salt):
    return hash(s + salt)% 1023

candidates = [23234, 8878, 8868]

def crack_code(u, salt, candidates):

    for x in candidates:
        
        u_c = hash_function(x, salt)
        if u == u_c:
            print("Found secret code for input hash: " + str(x))
            return True

    return False


def crack_code_and_salt(u, candidates):
    salt = 0

    while salt < 10:
        isSuccess = crack_code(u, salt, candidates)
        if isSuccess:
            print("secret salt is " + str(salt))
            return True

        salt = salt + 1

    print("Could not find code")
    return False


w = 90

s = 23234

secret_salt = 3

u = hash_function(s, secret_salt)

w = 90

crack_code_and_salt(u, candidates)


p1 = {123 : 'very sensitive data on person 123',
      555 : 'very sensitive data on person 555'}







w = 90


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