


x       r    g     b     a
1       1
3            1
5       2.1 
8                  1
9                         2


# Red

import numpy as np

x_all = np.array([1,3,5,8,9])

x_red = np.array([1, 5])
red = np.array([1, 2.1])

x_green = np.array([3])
green = np.array([1])

x_blue = np.array([8])
blue = np.array([1])

x_alpha = np.array([9])
alpha = np.array([2])

r_sequence = np.interp(x_all, x_red, red, left=None, right=None, period=None)
g_sequence = np.interp(x_all, x_green, green, left=None, right=None, period=None)
b_sequence = np.interp(x_all, x_blue, blue, left=None, right=None, period=None)
a_sequence = np.interp(x_all, x_alpha, alpha, left=None, right=None, period=None)

sequence = np.vstack([r_sequence, g_sequence, b_sequence, a_sequence])

# This sequence vector and x_all defines known points


# input: sample locations

sample_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

j = np.searchsorted(x_all, sample_x) - 1

d = (sample_x - x_all[j]) / (x_all[j + 1] - x_all[j])


# Outputs

sample_r = (1 - d) * sequence[0][j] + sequence[0][j + 1] * d
sample_g = (1 - d) * sequence[1][j] + sequence[1][j + 1] * d
sample_b = (1 - d) * sequence[2][j] + sequence[2][j + 1] * d
sample_a = (1 - d) * sequence[3][j] + sequence[3][j + 1] * d



