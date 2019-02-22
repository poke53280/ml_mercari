

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
cipher = AES.new(key, AES.MODE_CBC)


array_in = np.random.randint(0, 999999, 300 * 1000 * 1000)


array_in_base64_str = base64.b64encode(array_in)

padded_data = pad(array_in_base64_str, cipher.block_size)

txt_encrypted = cipher.encrypt(padded_data)

f = open("c:\\crypto_data\\output_numpy.bin", "wb")

f.write(txt_encrypted)

f.close()

# ...

f = open("c:\\crypto_data\\output_numpy.bin", "rb")

from_file = f.read()

f.close()


cipher2 = AES.new(key, AES.MODE_CBC, cipher.iv)

back = cipher2.decrypt(from_file)


unpadded_data = unpad(back, cipher.block_size)


r = base64.decodestring(unpadded_data)


q = np.frombuffer(r, dtype=np.int32)


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


f = open("c:\\crypto_data\\output_string.bint", "rb")

from_file = f.read()

f.close()

cipher2 = AES.new(key, AES.MODE_CBC, cipher.iv)


back = cipher2.decrypt(from_file)


unpadded_data = unpad(back, cipher.block_size)

r = base64.decodestring(unpadded_data)


s_txt_out = r.decode(encoding="utf-8")


assert s_txt_out == string_in







