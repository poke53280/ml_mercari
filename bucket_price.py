

import numpy as np
import pandas as pd

DATA_DIR = "C:\\Users\\T149900\\ml_mercari\\"

train = pd.read_table(DATA_DIR + "train.tsv");
test = pd.read_csv(DATA_DIR + "test.tsv", sep = "\t", encoding="utf-8", engine = "python");

full = train.append(test)


class Bucketer():
    def __init__(self, price_series):
        l = price_series.value_counts()
        a = zip(l, l.index)
        _, w = zip(*a)

        w = w[:90]

        self.base = np.sort(w)

        self.mid_base = self.base[1:] + self.base[:-1]
        self.mid_base = self.mid_base / 2

    def get_binned_price(self, my_nums):
        closest_indices = np.digitize(my_nums, self.mid_base)
        return self.base[closest_indices]


"""Class end"""


"""RUN"""

b = Bucketer(train.price)

my_nums = np.array([31.2,9.9,55.3,128.9])

my_nums2 = b.get_binned_price(my_nums)