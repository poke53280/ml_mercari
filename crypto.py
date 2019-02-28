

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
import numpy as np
import base64


def get_long_string():

    string_val = "x" * 10
    string_val2 = string_val * 100
    string_val3 = "aa" + string_val2 * 100
    string_val4 = "yy" + string_val3 * 100

    return "anders" + string_val4[:3] + " anders" + string_val4 + "b" + string_val4 + "xx" + string_val4
"""c"""


# numpy array




key = get_random_bytes(16)

iv = get_random_bytes(16)
cipher = AES.new(key, AES.MODE_CBC, iv = iv)


array_in = np.random.randint(0, 999999, 300 * 1000 * 1000)


array_in_bytes = array_in.tobytes()

padded_bytes = pad(array_in_bytes, cipher.block_size)

ct_bytes = cipher.encrypt(padded_bytes)

ct_txt = base64.b64encode(ct_bytes).decode('utf-8')
iv_txt = base64.b64encode(cipher.iv).decode('utf-8')

dtype_txt = array_in.dtype.name
shape_txt = array_in.shape


## ...

ct_bytes_out = base64.b64decode(ct_txt)
iv_bytes_out = base64.b64decode(iv_txt)




cipher2 = AES.new(key, AES.MODE_CBC, iv_bytes_out)

back = cipher2.decrypt(ct_bytes_out)


unpadded_data = unpad(back, cipher.block_size)


q = np.frombuffer(unpadded_data, dtype = dtype_txt).reshape(shape_txt)



assert (q == array_in).all()

########################################################################################
#
# String
#

key = get_random_bytes(16)
cipher = AES.new(key, AES.MODE_CBC)


string_in = get_long_string()

string_in_array = string_in.encode(encoding="utf-8")

r = base64.encodestring(string_in_array)

padded_data = pad(r, cipher.block_size)

txt_encrypted = cipher.encrypt(padded_data)

f = open("c:\\crypto_data\\output_string.bin", "wb")

f.write(txt_encrypted)

f.close()


# ...


f = open("c:\\crypto_data\\output_string.bin", "rb")

from_file = f.read()

f.close()

cipher2 = AES.new(key, AES.MODE_CBC, cipher.iv)


back = cipher2.decrypt(from_file)


unpadded_data = unpad(back, cipher.block_size)

r = base64.decodestring(unpadded_data)


s_txt_out = r.decode(encoding="utf-8")


s_txt_out == string_in



##################### PASSWORDS ##########

from Crypto.Random import get_random_bytes
import base64
import numpy as np
import getpass


# Creation:

########################################################################################
#
# create_mem_password
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


def create_txt_password():

    x = input("Enter three digit number 0-999:")
    x = int (x)

    password_b = create_mem_password(x)

    password_b32 = base64.b32encode(password_b)

    password_b32_txt_upper = password_b32.decode('utf-8')

    password_res = password_b32_txt_upper.lower()

    return password_res

def get_mem_password_from_txt_password(password_res):

    password_back_b32_txt_upper = password_res.upper()

    password_back_b32 = password_back_b32_txt_upper.encode('utf-8')

    password_back_b = base64.b32decode(password_back_b32)

    return password_back_b
    


def get_checksum_char (txt):

    ordrange = 1 + ord('z')- ord('a')

    chk_value = hash(txt) % ordrange
    
    checksum_ord = ord('a') + chk_value
    checksum_char = chr(checksum_ord)
    return checksum_char
"""c"""


def get_chunked_password_with_chksum(txt, num_tokens_each_word): 
    num_chunks = len (txt) // num_tokens_each_word + 1 * (len (txt) % num_tokens_each_word > 0)

    idx = np.arange(start = 0, stop = num_tokens_each_word * (num_chunks + 1), step = num_tokens_each_word)

    idx_lo = idx[:-1]
    idx_hi = idx[1:]

    t = zip (idx_lo, idx_hi)

    l_words = []

    for idx, (b, e) in enumerate(t):
        c = get_checksum_char(txt[b:e])
        l_words.append(str(idx) + ":" + txt[b:e] + c)
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
        
            x = getpass.getpass(f"word {iWord}")
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



