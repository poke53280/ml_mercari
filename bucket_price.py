

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

        w = w[:100]

        self.base = np.sort(w)

        self.mid_base = self.base[1:] + self.base[:-1]
        self.mid_base = self.mid_base / 2

    def get_binned_price(self, my_nums):
        closest_indices = np.digitize(my_nums, self.mid_base)
        return self.base[closest_indices]


"""Class end"""


"""RUN"""

b = Bucketer(train.price)

my_nums = np.array([31.2,9.9,55.3,128.9, 2.1])

my_nums2 = b.get_binned_price(my_nums)



l = s.value_counts()

import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py

a = l.index

a = a.values

a = np.sort(a)

plt.plot(a[:100])

plt.show()

l = l.sort_index()

plt.hist(a)

fig = plt.gcf()
plt.show()


l[:10].sum()


a = np.random.randint(10, size=100)
b = a + 1

rmsle_func(a, b)

def bottom_digitize(x):
    if x < 3.5:
        return 3
    elif x < 4.5:
        return 4
    elif x < 5.5:
        return 5
    elif x < 6.5:
        return 6
    elif x < 7.5:
        return 7
    elif x < 8.5:
        return 8
    elif x < 9.5:
        return 9
    else:
        return x

w = 90

price_series.apply(bottom_digitize)



l = train.price.value_counts()
# possible to clip here
sum = l.sum()
l.apply(lambda x: x/sum)

# input stream:
# sort
# let each group take from bottom.




