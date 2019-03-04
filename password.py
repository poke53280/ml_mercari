

from Crypto.Random import get_random_bytes
import numpy as np
import base64
import getpass



########################################################################################
#
#   create_mem_password
#

def create_mem_password(x):
    key = get_random_bytes(32)

    assert x >= 0 and x <= 999, "Need a number in range [0, 999]"

    b = bytearray(key)

    b[0] = (b[0] + x + b[3]) % 256
    b[3] = (b[3] + b[2] + x) % 256
    b[7] = (b[8] + 2 * b[4] + x) % 256
    b[31] = (b[31] + b[17] + 3 * x * b[29] + x) % 256

    key = bytes(b)
    return key
"""c"""

########################################################################################
#
#   create_txt_password
#

def create_txt_password(password_b):

    password_b32 = base64.b32encode(password_b)

    password_b32_txt_upper = password_b32.decode('utf-8')

    password_res = password_b32_txt_upper.lower()

    return password_res

########################################################################################
#
#   get_mem_password_from_txt_password
#

def get_mem_password_from_txt_password(password_res):

    password_back_b32_txt_upper = password_res.upper()

    password_back_b32 = password_back_b32_txt_upper.encode('utf-8')

    password_back_b = base64.b32decode(password_back_b32)

    return password_back_b

    
########################################################################################
#
#   get_checksum_char
#

def get_checksum_char (txt):

    ordrange = 1 + ord('z')- ord('a')

    chk_value = hash(txt) % ordrange
    
    checksum_ord = ord('a') + chk_value
    checksum_char = chr(checksum_ord)
    return checksum_char
"""c"""

########################################################################################
#
#   get_chunked_password_with_chksum
#

def get_chunked_password_with_chksum(txt, num_tokens_each_word): 
    num_chunks = len (txt) // num_tokens_each_word + 1 * (len (txt) % num_tokens_each_word > 0)

    idx = np.arange(start = 0, stop = num_tokens_each_word * (num_chunks + 1), step = num_tokens_each_word)

    idx_lo = idx[:-1]
    idx_hi = idx[1:]

    t = zip (idx_lo, idx_hi)

    l_words = []

    for idx, (b, e) in enumerate(t):
        c = get_checksum_char(txt[b:e])
        l_words.append(str(idx +1) + ":" + txt[b:e] + c)
    """c"""

    out_txt = " ".join(l_words)

    return out_txt

"""c"""

###############################################################
#
# get_txt_password_from_cmdline
#

def get_txt_password_from_cmdline():

    num_chunks = int (input("Number of pieces:"))

    l_password = []

    for iWord in range(num_chunks):

        isEnterMode = True

        while (isEnterMode):
        
            x = getpass.getpass(f"word {iWord +1}")
            if len(x) < 2:
                print(f"Word must have >= 2 letters.")
            else:

                c_entered = x[-1]
                raw_word = x[:-1]
                c = get_checksum_char(raw_word)
                if c == c_entered:
                    l_password.append(raw_word)
                    isEnterMode = False
                else:
                    print (f"Checksum error. Try again.")
        
    """c"""

    return "".join(l_password)

"""c"""

########################################################################################
#
#   generate_password_cmdline
#

def generate_password_cmdline():
    x = input("Enter three digit number 0-999:")
    x = int (x)

    print (get_chunked_password_with_chksum(create_txt_password(create_mem_password(x)), 7))


########################################################################################
#
#   get_mem_password_from_cmdline
#


def get_mem_password_from_cmdline():
    return get_mem_password_from_txt_password(get_txt_password_from_cmdline())


# 1:6aukzysz 2:5ti4i5nj 3:d7mqj5xl 4:bvp5ddng 5:g3hkclbe 6:bwced2yy 7:fhfx77hb 8:uva====o
generate_password_cmdline()

get_mem_password_from_cmdline()