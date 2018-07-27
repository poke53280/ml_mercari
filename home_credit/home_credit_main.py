

import time
from general.TimeSlotAllocator import slot_allocator
import pandas as pd
import numpy as np
import gc
from category_encoders import *
  
DATA_DIR_PORTABLE = "C:\\home_credit_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
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

#########################################################################################
#
#   SetObjectToCategorical
#

def SetObjectToCategorical(d):

    lRemoveCat = list (d.keys())

    for df in lRemoveCat:
        q = d[df]

        cols = q.columns
        num_cols = q._get_numeric_data().columns
        cat_columns = list(set(cols) - set(num_cols))

        for c in cat_columns:

            cat_series = q[c].astype('category')

            kwargs = {c : cat_series}

            q = q.assign(**kwargs)

        d[df] = q
    return d

##########################################################################################
#
#   BinaryEncodeAllCategoricalColumns
#

def BinaryEncodeAllCategoricalColumns(q):
    cols = q.columns
    num_cols = q._get_numeric_data().columns
    cat_columns = list(set(cols) - set(num_cols))

    enc = BinaryEncoder(cols=cat_columns)
    q_new = enc.fit_transform(q)

    print(f"   Done. Encoded {len(cat_columns)} categorical column(s). Num features {q.shape[1]} -> {q_new.shape[1]}")

    return q_new

"""c"""


start_time = time.time()
        
print('[{}] Go'.format(time.time() - start_time))

d = load_all()

# Note: Verify test/train fit/transform
d = SetObjectToCategorical(d)

gc.collect()

print('[{}] Data loaded'.format(time.time() - start_time))

y = d['train']['TARGET']
del d['train']['TARGET']



########## Group bureau days_credit by sk_id_curr ###################

train = d['train']
test = d['test']

l_user = d['train'].SK_ID_CURR.unique()
l_user2 = d['test'].SK_ID_CURR.unique()

an = np.concatenate([l_user, l_user2])


b = d['b']
q = b[['SK_ID_CURR', 'DAYS_CREDIT']]

q = q[q.SK_ID_CURR.isin(an)]


g = q.groupby(by = 'SK_ID_CURR')


e = g['DAYS_CREDIT'].apply(lambda x: x.tolist())


#verify:
m = q.SK_ID_CURR ==162297
q[m]

# equals:

e[162297]

def sort_values(x):
    x.sort()

e.apply(sort_values)



def get_element_value(x, idx):
    if len(x) >= -idx:
        return x[idx]
    else:
        return np.nan


l_slots = []

for slot in [0,1,2,3,4]:

    idx = -1 - slot
    hi = e.apply(get_element_value, args = (idx,))
    an_hi = np.array(hi)

    rSlotMean = np.nanmean(an_hi) 

    print(f"Slot #{slot}: Mean value = {rSlotMean}")
    l_slots.append(rSlotMean)



def designate_slots(x):
    x.sort(reverse = True)
    return slot_allocator(x, l_slots, 305, False)




q = e.apply(designate_slots)


q


def crit(x, threshold):
    return sum (i > threshold for i in x)

l_threshold = [-4000, -3000, -2000, -1500, -1000, -500]

for threshold in l_threshold:

    c = e.apply(crit, args = (threshold, ))

    ac = np.array(c)
    np.mean(ac)
    np.std(ac)

    print(f"With cut off at {threshold}, element count is {np.mean(ac)} +/- {np.std(ac)}")


# Todo: Fit - not fit_transform on train. Careful handling of unseen.

#for x in lRemoveCat:
#    print(f"Binary encoding '{x}'...")
#    d[x] = BinaryEncodeAllCategoricalColumns(d[x])
#
#"""c"""



num_train = d['train'].shape[0]
num_test = d['test'].shape[0]

num_users = num_train + num_test


num_features = 0

num_features_base = d['train'].shape[1] -1 

num_features += num_features_base

num_features_prev_app = d['prev'].shape[1]

num_prev_app = d['prev'].shape[0]

num_prev_app_per_user_mean = num_prev_app / num_users

print(f"Number of previous apps per user mean {num_prev_app_per_user_mean:.1f}")

num_featurs_from_previous_apps = num_prev_app_per_user_mean * num_features_prev_app

print(f"Number of features from previous apps: {num_featurs_from_previous_apps:.1f}")

num_features += (num_featurs_from_previous_apps)


# CCB

num_ccb = d['ccb'].shape[0]

num_features_ccb = d['ccb'].shape[1]

num_ccb_per_user_mean = num_ccb / num_users

num_features_from_ccb = num_ccb_per_user_mean * num_features_ccb

num_features += (num_features_from_ccb)

# INSTALMENT

num_inst = d['ip'].shape[0]

num_features_inst = d['ip'].shape[1]

num_inst_per_user_mean = num_inst / num_users

num_features_from_inst = num_inst_per_user_mean * num_features_inst

num_features += (num_features_from_inst)

# POS CASH BALANCE

num_pcb = d['pcb'].shape[0]

num_features_pcb = d['pcb'].shape[1]

num_pcb_per_user_mean = num_pcb / num_users

num_features_from_pcb = num_pcb_per_user_mean * num_features_pcb

num_features += num_features_from_pcb


# BUREAU

num_bur = d['b'].shape[0]

num_features_bur = d['b'].shape[1]

num_bur_per_user_mean = num_bur / num_users

num_features_from_bur = num_bur_per_user_mean * num_features_bur

num_features += num_features_from_bur

# BUREAU BALANCE

num_bb = d['bb'].shape[0]

num_features_bb = d['bb'].shape[1]

num_bb_per_user_mean = num_bb / num_users

num_features_from_bb = num_bb_per_user_mean * num_features_bb

num_features += num_features_from_bb

print(f" Feature number estimate: {num_features}")




g_Cat = {}

d['bb']["STATUS"] = d['bb'].STATUS.astype('category')
g_Cat["BureauBalance_Status"] = d['bb'].STATUS.cat.categories.values
g_Cat["Bureau_CreditActive"] = d['b'].CREDIT_ACTIVE.cat.categories.values
g_Cat["Bureau_CreditCurrency"] = d['b'].CREDIT_CURRENCY.cat.categories.values
g_Cat["Bureau_CreditType"] = d['b'].CREDIT_TYPE.cat.categories.values




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

    
def detail_loan_bureau(id):
    q = get_bureau_loan_details(id)

    if len(q) == 0:
        return {}

    anbur = np.array(q.SK_ID_BUREAU)

    assert(anbur.min() == anbur.max())
    assert(anbur.min() == id)

    anMonth = np.array(q.MONTHS_BALANCE)

    anState = np.array(q.STATUS)

    bdetails = {}

    for i in range(0,len(anMonth)):
        month = anMonth[i]
        bdetails[month] = anState[i]

    return bdetails

"""c"""



def describe_bureau_loan(q_loan, dict_detail):
    
    print(f" {q_loan['CREDIT_ACTIVE']} - {q_loan['CREDIT_TYPE']}")
    
    if len(dict_detail) > 0:
        print(dict_detail)
    else:
        print("   No detail")

"""c"""

class BureauLoan:
    def __init__(self, qLocal):

        sk_id_curr = qLocal.SK_ID_CURR.values[0]
        b_id = qLocal.SK_ID_BUREAU.values[0]

        this_currency = qLocal.CREDIT_CURRENCY.values[0]

        # Drop currency. Could have been kept as a signature property, but there are very few non 'currency 1's.

        self._dict_detail = detail_loan_bureau(b_id)
 
        # Status of the Credit Bureau (CB) reported credits,
       
        # Closed      1079273
        # Active       630607
        # Sold           6527
        # Bad debt         21

        self._credit_active = qLocal.CREDIT_ACTIVE.cat.codes.values[0]

        # CREDIT TYPE
        # Consumer credit                                 1251615
        # Credit card                                      402195
        # Car loan                                          27690
        # Mortgage                                          18391
        # Microloan                                         12413
        # Loan for business development                      1975
        # Another type of loan                               1017
        # Unknown type of loan                                555
        # Loan for working capital replenishment              469
        # Cash loan (non-earmarked)                            56
        # Real estate loan                                     27
        # Loan for the purchase of equipment                   19
        # Loan for purchase of shares (margin lending)          4
        # Mobile operator loan                                  1
        # Interbank credit                                      1

        self._credit_type = qLocal.CREDIT_TYPE.cat.codes.values[0]

        # Number of days past due on CB credit at the time of application for related loan in our sample,
        self._credit_day_overdue = qLocal.CREDIT_DAY_OVERDUE.values[0]

        # How many times was the Credit Bureau credit prolonged,
        self._cnt_credit_prolong = qLocal.CNT_CREDIT_PROLONG.values[0]
        
        # TIME ONLY RELATIVE TO THE APPLICATION IN DAYS

        # How long before current application did client apply for Credit Bureau credit
        self._days_credit = qLocal.DAYS_CREDIT.values[0]

        # Remaining duration of CB credit 
        self._days_credit_enddate = qLocal.DAYS_CREDIT_ENDDATE.values[0]
        
        # Time since CB credit ended at the time of application in Home Credit (only for closed credit)
        self._days_enddate_fact = qLocal.DAYS_ENDDATE_FACT.values[0]

        # How long before loan application did last information about the Credit Bureau credit come (?)
        self._days_credit_update = qLocal.DAYS_CREDIT_UPDATE.values[0]

        # Current amounts.... at application date of loan

        #      Maximal overdue on the Credit Bureau credit (so far)
        self._amt_credit_max_overdue = qLocal.AMT_CREDIT_MAX_OVERDUE.values[0]
        
        #       Credit amount for the Credit Bureau credit,
        self._amt_credit_sum = qLocal.AMT_CREDIT_SUM.values[0]

        #       Debt on Credit Bureau credit,
        self._amt_credit_sum_debt = qLocal.AMT_CREDIT_SUM_DEBT.values[0]

        #       Credit limit of credit card reported in Credit Bureau,
        self._amt_credit_sum_limit = qLocal.AMT_CREDIT_SUM_LIMIT.values[0]

        #       Amount overdue on Credit Bureau credit,
        self._amt_credit_sum_overdue = qLocal.AMT_CREDIT_SUM_OVERDUE.values[0]

        # Annuity of the Credit Bureau credit,
        self._amt_annuity = qLocal.AMT_ANNUITY.values[0]

    def GetCreditActiveDesc(self):
        desc = g_Cat["Bureau_CreditActive"][self._credit_active]
        return desc

    def GetCreditTypeDesc(self):
        desc = g_Cat["Bureau_CreditType"][self._credit_type]
        return desc

    def Desc(self):
        print("-------------------------------------------")
        print(self._dict_detail)
        
        print(f"Credit type: {self.GetCreditTypeDesc()}")
        print(f"Credit active: {self.GetCreditActiveDesc()}")
       
        print(f"Application time wrt now: {self._days_credit}")
       
        print(f"Days overdue now: {self._credit_day_overdue}")
        print(f"Day still valid now: {self._days_credit_enddate}")
        print(f"Days since ended wrt now: {self._days_enddate_fact}")
        print(f"Days since last update wrt now: {self._days_credit_update}")
       
        print(f"_amt_credit_sum: {self._amt_credit_sum}")
        print(f"_amt_credit_sum_debt: {self._amt_credit_sum_debt}")
        print(f"_amt_credit_sum_limit: {self._amt_credit_sum_limit}")
        print(f"_amt_credit_sum_overdue: {self._amt_credit_sum_overdue}")
        print(f"_amt_credit_max_overdue: {self._amt_credit_max_overdue}")
       
        print(f"_amt_annuity: {self._amt_annuity}")
        print(f"_cnt_credit_prolong: {self._cnt_credit_prolong}")
"""c"""

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

    # See: https://www.kaggle.com/c/home-credit-default-risk/discussion/57562

    if num_pos_cash_balance_user != num_pcb_prev_sum:
       pass

    if num_inst_pay_user != num_inst_prev_sum:
       pass

    if num_cc_balance_user != num_cc_balance_sum:
       pass

    ############## BUREAU CREDIT REPORTED LOANS ###############################

    q_b = get_on_sk_id_curr(user_id, 'b')

    print(f"Credit Bureau reported loan(s): {len(q_b)}")

    b_ids = list (q_b.SK_ID_BUREAU)

    # b_ids = [6353318]
    
    for b_id in b_ids:
        m = q_b.SK_ID_BUREAU == b_id
        q_loan = q_b[m]

        b = BureauLoan(q_loan)
        b.Desc()


"""c"""


all_users = np.array(d['train'].SK_ID_CURR) 

sampled_users = np.random.choice(all_users, replace= False, size = 10)

for id in sampled_users:
    print(f"------------Checking user {id}...")
    get_user_data(id)

"""c"""



get_user_data(370747)

"""c"""

# Start work on current application


#
#
# Prepare as feature: 
#    float w.nan.
#    integer.
#    Category.
#
# Category to binary label
#
#
# Initialization:
#
#
##----app curr-------------- app prev slot ------------------ app prev slot ---------------- app prev slot --------------------bureau slot--------------bureau slot-------------bureau slot-----------


id = 215354

q = d['b']

m = q.SK_ID_CURR == id

q = q[m]

len (q)

q.shape

# To numpy
X = np.array(q)

X.shape


############################################################################################################
#
# Checking out combining many rows into one row.
#
#

data = np.array([['','ID', 'Col1','Col2'], ['Row1',0, 1, 2], ['Row2',0, 3,4], ['Row3',1, 7,9]])
                 
df = pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:])

X_in = np.array(df, dtype = "float32")

X_res = np.empty(shape=(10), dtype="float32")
X_res[:] = np.nan
 

####################################################################
#           
#                       CopyRow
#
# Don't copy first num_skip elements, 0 to copy full row

def CopyRow(X_in, X_res, sourceRow, destOffset, num_skip):
    nColumns_in = X_in.shape[1] -num_skip
    X_res[destOffset: destOffset + nColumns_in] = X_in[sourceRow][num_skip:]

"""c"""

CopyRow(X_in, X_res, 0, 4, 0)


def test_report_size_Self():
    return 3


def test_Add_Self(X_res, offset):
    num_A = 90
    num_B = 11
    num_C = 92

    X_in = np.empty(shape=(1, 3), dtype="float32")

    X_in[0] = [num_A, num_B, num_C]

    CopyRow(X_in, X_res, 0, offset, 0)

"""c"""

#################################################################
#
# Visualize spacing on prev. distribution.
#
# Info spacing in time
#
# slots... exact... count...A
#
# find good compromise...
#
#
# distribution - num prevs since app time on day
#
#

nApps = len (d['train'])+ len (d['test'])

print(f"# applications total: {nApps}")

q = d['prev']

nAppsInPrev = q.SK_ID_CURR.nunique()
print(f"# applications in prev: {nAppsInPrev}")

nPrev_Per_curr_Mean = len(q) / nApps

freq = q.SK_ID_CURR.value_counts()

s2 = set (d['train'].SK_ID_CURR)
s3 = set (d['test'].SK_ID_CURR)

s4 = s2.union(s3)

print(f"Number of SK_ID_CURR in user set: {len(s4)}")


s_prev = set(freq.index)

print(f"Number of SK_ID_CURR in prev set: {len(s_prev)}")

# Same:
s_int = s4.intersection(s_prev)

print(f"SK_ID_CURR intersection prev and curr: {len(s_int)}")

# Numbers show that all SK_ID_CURR in prev set are in user set

l = list (freq.values)

nNoPrev = nApps - len (s_int)

l.sum()

# List length - showing number of prev apps for each curr app + curr apps with no prev apps - equal all apps)
assert (len (l) + nNoPrev == nApps)

from matplotlib import pyplot as plt

plt.hist(l, bins = 50)

plt.show()

np.array(l).min()

# Cut off at 10 - 15 - 20. Try 20 for

q = q[['SK_ID_CURR', 'DAYS_DECISION']]

an = np.arange(0, 3500, 1)

x = []
y = []

for r in an:
    
    m = q.DAYS_DECISION > -r
    p = q[m]

    nTotal = len(q)
    nAcc   = len(p)

    fFac = 100.0 * nAcc/nTotal 

    # print(f"day desc > {r}: {nAcc} prev apps of in total {nTotal} apps. {fFac:.2f}%")
    x.append(r)
    y.append(fFac)

"""c"""


import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.show()


#
# Test (from Train_merge.py)
#


#############################################################################################
#
#  Working on bureau balance
#


d = load_all()

# Process bureau balance

q = d['bb']

q.isnull().sum()

q.STATUS.value_counts()

q.STATUS = q.STATUS.astype('category')

l_cat_status = q.STATUS.cat.categories.values

q.STATUS = q.STATUS.cat.codes

q.MONTHS_BALANCE = pd.to_numeric(q.MONTHS_BALANCE, downcast = 'signed')

q.dtypes

q_balance = q[['SK_ID_BUREAU', 'MONTHS_BALANCE']]

keys, values_balance = q_balance.sort_values('SK_ID_BUREAU').values.T

q_status = q[['SK_ID_BUREAU', 'STATUS']]

keys_s, values_status = q_status.sort_values('SK_ID_BUREAU').values.T


ukeys,index=np.unique(keys, return_index = True)

arrays_balance = np.split(values_balance,index[1:])
arrays_status = np.split(values_status, index[1:])

assert (len (ukeys) == len (arrays_balance))
assert (len (ukeys) == len (arrays_status))

#########################################################
#
#   is_dense()
#

def is_dense(a):

    if (len(a) != len(np.unique(a))):
        print("Duplicates found")
        return

    aMin = a.min()
    aMax = a.max()

    aDiff = aMax - aMin

    isDense = (aDiff == len(a) -1)

    return isDense

"""c"""

nDense = 0
nSparse = 0

for a in arrays_balance:
    if is_dense(a):
        nDense = nDense + 1
    else:
        nSparse = nSparse + 1

"""c"""

print(f"Sparse: {nSparse}, Dense: {nDense}")


#
#
# => All are dense
#
#
# Todo: Produce:
# 
#   SK_ID_BUREAU Start_Month  Length   Status_String
#
# Todo: Check characteristics of status string - consider feature engineering
#
#
############################################################################
#
#Time slot to a certain degree
#
#App 1        App 2         App 3        App 4
#
#

import numpy as np


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idxfull_slots 


#
#
# Data analyis - placement in time.
#
#
# Data   Property
# Data   Time related : 'x'
# Data   Time related : 'x'
# Data   Time related : 'x'
# Data   Time related : 'x'
# Data   Time related : 'x'
# Data   Time related : 'x'
# Data   Property. Category
# Data  Time related start
# Data  Time related stop
#
# 



##################################################################################################
#
#    TimeData
#
#




class TimeData:

    # Base time - T0
    _t0 = 0

    # Header. Fixed amount of data
    _a = 9.2
    _b = 12.9
    _c = 19.9
    _d = 21


    # Time line data. Varying in length. Fixed amount per time slot

    _l = [[1,2,3], [-1, -1, 3], [3,3,3]]
    _ts0 = 7
    

    def __init__(self, t0):
        self._t0 = t0

    def getT0(self):
        return self._t0

    def getTimespan(self):
        ts1 = self._t0 + len (self._l)

    def getBaseSize(self):
        return 4   #_a, _b, _c, _d

    def getIntervalSize(self):
        return 3


"""c"""

t = TimeData(11)

t.getT0()






t0 = []

t0.append(TimeData(110))
t0.append(TimeData(90))
t0.append(TimeData(50))
t0.append(TimeData(30))
t0.append(TimeData(10))

t_slots = [120, 100, 90, 40, 20, 0]

tolerance = 5


t0_data = []

for t in t0:
    t0_data.append(t.getT0())

     

filled_slot = slot_allocator(t0_data, t_slots, 10, False)


num_slots = len(t_slots)

datasize = 4

avMem = np.empty( (num_slots, datasize), dtype = np.float32)

filled_slot = assign_to_slots(t0_data, t_slots, tolerance)

for i, s in enumerate(filled_slot):
    if s == -1:
        print(f"slot {i}, target value {t_slots[i]}: Empty")
        avMem[i] = 0
    else:
        mem = t0[s].fit_to_slot(t_slots[i], datasize)
        assert len(mem) == datasize
        avMem[i] = mem

avMem

avMem = avMem.reshape(datasize * num_slots )


#
# ccb with home credit.
#
# Some SK_ID_CURR  100006, 456233, 456243, 456254
#
#

class CreditCardBalanceRecord:
    
    # An amount of data per time slot
    _balance = 0
    
    def __init__(self, s):
        print(f"Constructing CCB object, amount is {s.AMT_BALANCE}")
        self._balance = s.AMT_BALANCE

"""c"""

class CreditCardBalance:

    _records = {}

    def __init__(self, q):

        for i in range(len(q)):
            s = q.iloc[i]
            time = s.MONTHS_BALANCE
            rec = CreditCardBalanceRecord(s)
            assert not (time in self._records)
            self._records[time] = rec

    def getTimeStamps(self):
        l = list (self._records.keys())
        return list (np.sort(np.array(l))[::-1])
"""c"""

from general.TimeSlotAllocator import slot_allocator



q = get_on_sk_id_curr([456233], 'ccb')

c = CreditCardBalance(q)

list_time_slots_configuration = [-1,-2, -17]

list_time_values = c.getTimeStamps()

l_slot_allocation = slot_allocator(list_time_values, list_time_slots_configuration, 0.0, False)


for i, alloc_location in enumerate(l_slot_allocation):
    print(f"slot {i}, requesting data index = {alloc_location}")



list_time_values