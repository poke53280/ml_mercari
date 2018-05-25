
import time
import pandas as pd
import numpy as np
import gc

  
DATA_DIR_PORTABLE = "C:\\home_credit_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE


def load_frame(name):
    print(f"Loading {name}...")
    return pd.read_table(DATA_DIR + name + ".csv", sep = ",");

"""c"""

def load_all():

    d = {}

    file_info = {'application_train': 'train',
                 'application_test' : 'test' ,
                 'bureau'           : 'b',
                 'bureau_balance'   : 'bb',
                 'credit_card_balance' : 'ccb',
                 'installments_payments' : 'ip',
                 'POS_CASH_balance' : 'pcb',
                 'previous_application' : 'prev'}


    for k,v in file_info.items():
   
        df = load_frame(k)
        d[v] = df

    return d

"""c"""


start_time = time.time()
        
print('[{}] Go'.format(time.time() - start_time))


d = load_all()

gc.collect()

print('[{}] Data loaded'.format(time.time() - start_time))






