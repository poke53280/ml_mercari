

DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"

DATA_DIR_BASEMENT = "D:\\mercari\\"

DATA_DIR = DATA_DIR_BASEMENT


import re
import pandas as pd

def analyze_run_data_1():
    f = open(DATA_DIR + "rundata.txt")
    s = f.read()
    f.close()
    rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)

    new_list = [rr[i:i+5] for i in range(0, len(rr), 5)]
  
    df = pd.DataFrame(new_list, columns = ['max_bin', 'num_leaves', 'learing_rate', 'RMSE', 'time'])
    df = df.convert_objects(convert_numeric = True)

    

    import seaborn as sns

    sns.set(style="ticks", color_codes = True)

    sns.set()

    g = sns.lmplot(x = "time", y ="RMSE", hue="num_leaves", truncate = True, size = 5, data=df)

