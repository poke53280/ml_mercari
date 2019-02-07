

import pickle
import numpy
from simplecrypt import encrypt, decrypt


password = 'sekret'


a = numpy.array([3,4,5,61])

serialized = pickle.dumps(a, protocol=0)
ciphertext = encrypt(password, serialized)

back_again = decrypt(password, ciphertext)

a_back = pickle.loads(back_again)
a_back




