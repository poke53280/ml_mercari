

import numpy as np

a0 = np.random.choice(1684 * 1920 *3 , size = 100)
a1 = np.random.choice(1684 * 1920 *3, size = 100)


hash(bytes(a0))
hash(bytes(a1))