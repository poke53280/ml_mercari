

DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"

DATA_DIR_BASEMENT = "D:\\mercari\\"

DATA_DIR = DATA_DIR_BASEMENT


import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def error_contrib(a, low, size):
    acc = ((a >= low) & (a < low + size)).sum()
    contrib = (low + .5 * size) * acc

    return { 'contrib' : contrib, 'count' : acc }

w = 90

def error_dist(a, size):
    low = 0.0
    acc = 0.0
    max = a.max()

    while low < max:
        d = error_contrib(a, low, size)
        error = d['contrib']
        count = d['count']

        acc = acc + error

        if error > 0:
            print("low = " + str(low) + ", e_acc= "+ str(error) + ", count=" + str(count))
        low = low + size


    n = len(a)

    mean = acc/n

    rmsle_acc = np.sqrt(mean)
    rmsle_full = np.sqrt(np.mean(a))
    
    print("rmsle real = " + str(rmsle_full) + " rmsle binned=" + str(rmsle_acc))


w = 90


def plot_prediction_error(a):
    g = sns.distplot(n2, bins = 1000)
    plt.show()


w = 90

def analyze_run_data_1():
    f = open(DATA_DIR + "rundata.txt")
    s = f.read()
    f.close()
    rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)

    new_list = [rr[i:i+5] for i in range(0, len(rr), 5)]
  
    df = pd.DataFrame(new_list, columns = ['max_bin', 'num_leaves', 'learing_rate', 'RMSE', 'time'])
    df = df.convert_objects(convert_numeric = True)

    

   

    sns.set(style="ticks", color_codes = True)

    sns.set()

    g = sns.lmplot(x = "time", y ="RMSE", hue="num_leaves", truncate = True, size = 5, data=df)

