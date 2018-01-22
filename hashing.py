
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

w = 90

SSN = [23234, 8878, 8868]

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

# Find 5 digit personal code given birth code by brute forcing

def find_SN(u, salt):
    candidate_start = 8878

    attempts = 100

    while attempts > 0:

        candidate = candidate_start + attempts

        u_c = hash_function(candidate, salt)

        if (u_c == u):
            print("Found candidate " + str(candidate))
            return True

        attempts = attempts -1

    return False

w = 90        



#With day of birth, find full SN by brute force until hash found:

p0 = 8901

s0 = 3
u0 = hash_function(p0, s0)

d0 = { 'hash': u0, 'salt': s0 }


#Store hash and salt for person ID.

p1 = 9903

s1 = 7
u1 = hash_function(p1, s1)

d1 =  { 'hash': u1, 'salt': s1 }


l = [d0, d1]

# Got list of hash and salt. Test to see if a group of people resolve

p = [8901, 9021, 1312]

for id in p:
    for d in l:
        _hash = d['hash']
        _salt = d['salt']

        hash_attempt = hash_function(id, _salt)

        if _hash == hash_attempt:
            print("Found given person " + str(id) + " in list")







w = 90

crack_code_and_salt(u, candidates)







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