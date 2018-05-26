
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

d.keys()


def get_on_sk_id_curr(id_list, db_name):
    m = (d[db_name].SK_ID_CURR.isin(id_list))

    q = d[db_name][m]
    print(f"Records in {db_name}: {len(q)}")

    return q

def get_on_sk_id_prev(id_list, id_curr, db_name):
    m = (d[db_name].SK_ID_PREV.isin(id_list))

    q = d[db_name][m]
    print(f"Records in {db_name}: {len(q)}")

    if len(q) > 0:
        m = (q.SK_ID_CURR == id_curr)
        assert (m.value_counts()[True] == len (q))

    return q

def get_bureau_loan_details(bureau_loan_id):
    m = (d['bb'].SK_ID_BUREAU == bureau_loan_id)

    q = d['bb'][m]

    return q

"""c"""


q_b = d['b']
q_bb = d['bb']

nFac = q_bb.SK_ID_BUREAU.nunique() / q_b.SK_ID_BUREAU.nunique()

print(f"Bureau loans with balance details: {100.0 * nFac}")


#### INFO ON ONE USER



def get_user_data(id):

    user_id = [id]

    q_train = get_on_sk_id_curr(user_id, 'train')
    assert(len(q_train) == 1)

    q_test = get_on_sk_id_curr(user_id, 'test')
    assert (len(q_test) == 0)

    ############## PREVIOUS APPLICATION ##############################

    q_prev = get_on_sk_id_curr(user_id, 'prev')
    prev_ids = list (q_prev.SK_ID_PREV)

    print(f"{len(prev_ids)} previous application(s)")

    # Pos cash balance

    num_pos_cash_balance_user = len (get_on_sk_id_curr(user_id, 'pcb'))
    num_pcb_prev_sum = 0

    # Instalments_payments

    num_inst_pay_user = len (get_on_sk_id_curr(user_id, 'ip'))
    num_inst_prev_sum = 0

    # Credit card balance

    num_cc_balance_user = len (get_on_sk_id_curr(user_id, 'ccb'))
    num_cc_balance_sum = 0

    for p_id in prev_ids:
    
        print(f"{user_id[0]}: Checking previous application: {p_id}")
    
        q = q_prev[q_prev.SK_ID_PREV == p_id]

        assert (len(q) == 1)

        # Pos cash balance
        q_pcb = get_on_sk_id_prev([p_id], user_id[0], 'pcb')
        num_pcb_prev_sum = num_pcb_prev_sum + len (q_pcb)

        # Installment 
        q_inst = get_on_sk_id_prev([p_id], user_id[0], 'ip')
        num_inst_prev_sum = num_inst_prev_sum + len (q_inst)

        # cc balance
        q_ccb = get_on_sk_id_prev([p_id], user_id[0], 'ccb')
        num_cc_balance_sum = num_cc_balance_sum + len (q_ccb)

    if num_pos_cash_balance_user != num_pcb_prev_sum:
        print(f"WARNING: pos cash balance user all: {num_pos_cash_balance_user} vs sum prev: {num_pcb_prev_sum} ")

    if num_inst_pay_user != num_inst_prev_sum:
        print(f"WARNING: inst pay records user all: {num_inst_pay_user} vs sum prev: {num_inst_prev_sum} ")

    if num_cc_balance_user != num_cc_balance_sum:
        print(f"WARNING: cc balance user all: {num_cc_balance_user} vs sum prev: {num_cc_balance_sum} ")

    ############## BUREAU CREDIT REPORTED LOANS ###############################

    q_b = get_on_sk_id_curr(user_id, 'b')

    print(f"Credit Bureau reported loan(s): {len(q_b)}")

    b_ids = list (q_b.SK_ID_BUREAU)

    for b_id in b_ids:
        q = get_bureau_loan_details(b_id)
        print(f"Loan detail records: {len(q)}")

"""c"""

all_users = np.array(d['train'].SK_ID_CURR)

sampled_users = np.random.choice(all_users, replace= False, size = 9)

for id in sampled_users:
    print(f"------------Checking user {id}...")
    get_user_data(id)

"""c"""


user_id
408584

two previous:
408584: Checking previous application: 2150911
408584: Checking previous application: 1091979

pos cash balance count  14

408584: Checking previous application: 2150911
Records in pcb: 0
Records in ip: 0
Records in ccb: 0
408584: Checking previous application: 1091979
Records in pcb: 11
Records in ip: 11
Records in ccb: 0

WARNING: pos cash balance user all: 14 vs sum prev: 11 
WARNING: inst pay records user all: 80 vs sum prev: 11 


