



from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad
from Cryptodome.Util.Padding import unpad
import numpy as np
import base64
import getpass

import pandas as pd
from pathlib import Path
import hashlib
import pickle

import string
import random
import array

import re

# Never ever set this to anything but None
g_PASSWORD = None


# (8 * 32) / np.log2(num_ascii)

#"=> 55 letters"


def save_encrypted(df, zPassword, output_file):

    key = hashlib.sha256(zPassword.encode()).digest()


    iv = get_random_bytes(16)

    # Encrypt array
    cipher = AES.new(key, AES.MODE_CBC, iv = iv)

    array_in_bytes = pickle.dumps(df)

    padded_bytes = pad(array_in_bytes, cipher.block_size)

    ct_bytes = cipher.encrypt(padded_bytes)

    ct_txt = base64.b64encode(ct_bytes).decode('utf-8')
    iv_txt = base64.b64encode(cipher.iv).decode('utf-8')

    # Array encrypted and encoded. Store.
    print(f"{ct_txt[:30]}...{ct_txt[-30:]}")

    encrypted_data = [ct_txt, iv_txt]

    pickle.dump(encrypted_data, open( output_file, "wb" ))


def load_encrypted(zPassword, input_file):

    key = hashlib.sha256(zPassword.encode()).digest()

    assert input_file.is_file()

    encrypted_data = pickle.load(open(input_file, "rb"))

    ct_txt = encrypted_data[0]
    iv_txt = encrypted_data[1]

    ct_bytes_out = base64.b64decode(ct_txt)
    iv_bytes_out = base64.b64decode(iv_txt)

    cipher2 = AES.new(key, AES.MODE_CBC, iv_bytes_out)

    back = cipher2.decrypt(ct_bytes_out)

    unpadded_data = unpad(back, cipher2.block_size)

    df = pickle.loads(unpadded_data)

    return df

"""c"""

####################################################################################
#
#   generate_passchunk
#

def generate_passchunk():

    l = np.random.randint(ord('a'), ord('z') + 1, 8)

    passchunk = array.array('B', l).tobytes().decode("utf-8")

    return passchunk


####################################################################################
#
#   check_input
#

def check_input(s):

    if len (re.findall(r'[a-z]', s)) != len(s):
        print("All chars must be A-Z")
        return False

    if len(s) != 9:
        print("Input must be 9 characters long")
        return False

    return True


####################################################################################
#
#   get_checksum_char
#

def get_checksum_char(passchunk):
    assert len(passchunk) == 8

    num_ascii = 1 + ord('z') - ord ('a')

    hash_remainder = (hashlib.sha256(passchunk.encode()).digest())[0] % num_ascii

    checksum_char = chr(ord('a') + hash_remainder)

    assert checksum_char <= 'z'
    assert checksum_char >= 'a'

    return checksum_char


####################################################################################
#
#   enter_password
#

def enter_password():

    l_chunks = []
    iChunk = 0

    while iChunk < 7:

        s = input(f"{iChunk + 1}/7: ")

        s = s.lower()
        s = s.replace(' ', '')

        if s == 'abort':
            break

        isVerified = check_input(s)

        if not isVerified:
            continue

        passchunk = s[:8]
        checksum = s[8]

        if get_checksum_char(passchunk) != checksum:
            print("Checksum error")
            continue

        l_chunks.append(s)
        iChunk = iChunk + 1

    return l_chunks


####################################################################################
#
#   format_chunk_line
#

def format_chunk_line(chunk, check):

    assert len(chunk) == 8
    assert len(check) == 1

    line = chunk + check

    line = line[0:3] + " " + line[3:6] + " " + line[6:]

    line = line.upper()

    return line


####################################################################################
#
#   generate_password_card
#

def generate_password_card():

    l_line = []

    for iLine in range(7):

        chunk = generate_passchunk()
        check = get_checksum_char(chunk)
        zLine = f"{iLine + 1}:" + format_chunk_line(chunk, check)

        l_line.append(zLine)

    return l_line


####################################################################################
#
#   print_password_card
#

def print_password_card(l_line):

    for iLine in range(len(l_line)):
        print(l_line[iLine])



####################################################################################
#
#   get_password_from_chunks
#

def get_password_from_chunks(l_chunks):

    l = [x[:8] for x in l_chunks]

    zPassword = "".join(l)

    assert len(zPassword) == 56

    return zPassword




# 1:RJN OAG TLU
# 2:XLV GLB ZBQ
# 3:MXM IYX XSI
# 4:NPX ZYT NGD
# 5:EQY VUP IMA
# 6:KJO TQL XZL
# 7:AAS VKS QQO





if __name__ == '__main__':

#l_line = generate_password_card()
# print_password_card(l_line)
    print("Case insensitive. Omit or include spaces at will. Give up/end password entering by 'ABORT' + Enter")
    l_chunks = enter_password()

    g_PASSWORD = get_password_from_chunks(l_chunks)

    #filepath = Path("C:\\Users\\T149900\\Downloads\\sp_count\\sp_count.pkl")

    #assert filepath.is_file()

    #df_0 = pd.read_pickle(filepath)


    output_file = Path("C:\\Users\\T149900\\Downloads\\sp_count\\sp_count_encrypted.pkl")

    #save_encrypted(df_0, g_PASSWORD, output_file)

    df_1 = load_encrypted(g_PASSWORD, output_file)

    print (df_1)

    #assert df_0.equals(df_1)




