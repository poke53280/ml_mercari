

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


