



import matplotlib.pyplot as plt

l_r = [0.000000, 0.007616, 0.007404, 0.000000, 0.000000, 0.039518, 0.047452, 0.063478, 0.051101, 0.047719, 0.000000, 0.000000, 0.109890,
       0.150980, 0.223048, 0.312769, 0.484874, 0.423759, 0.528706, 0.526603, 0.538025, 0.500737, 0.440351, 0.390866]


num_days = len (l_r)

l_d = list (range(num_days))


plt.plot(l_r)
plt.show()