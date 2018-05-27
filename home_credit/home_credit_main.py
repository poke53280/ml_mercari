
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

g_Cat = {}

d['bb']["STATUS"] = d['bb'].STATUS.astype('category')
g_Cat["BureauBalance_Status"] = d['bb'].STATUS.cat.categories.values

d['b']["CREDIT_ACTIVE"] =   d['b'].CREDIT_ACTIVE.astype('category')
g_Cat["Bureau_CreditActive"] = d['b'].CREDIT_ACTIVE.cat.categories.values

d['b']["CREDIT_CURRENCY"] = d['b'].CREDIT_CURRENCY.astype('category')
g_Cat["Bureau_CreditCurrency"] = d['b'].CREDIT_CURRENCY.cat.categories.values

d['b']["CREDIT_TYPE"] =     d['b'].CREDIT_TYPE.astype('category')
g_Cat["Bureau_CreditType"] = d['b'].CREDIT_TYPE.cat.categories.values


gc.collect()

g_currency_converter = get_bureau_currency_conversion(d)


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

def get_bureau_currency_conversion(d):
    q = d['b']

    f = {}

    m1 = q.CREDIT_CURRENCY == 'currency 1'
    m2 = q.CREDIT_CURRENCY == 'currency 2'
    m3 = q.CREDIT_CURRENCY == 'currency 3'
    m4 = q.CREDIT_CURRENCY == 'currency 4'

    v1 = q[m1].AMT_CREDIT_SUM.mean()
    v2 = q[m2].AMT_CREDIT_SUM.mean()
    v3 = q[m3].AMT_CREDIT_SUM.mean()
    v4 = q[m4].AMT_CREDIT_SUM.mean()

    f['currency 1'] = 1.0
    f['currency 2'] = v1/v2
    f['currency 3'] = v1/v3
    f['currency 4'] = v1/v4

    return f

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

        # Recoded currency of the Credit Bureau credit,recoded
        # currency 1    1715020
        # currency 2       1224
        # currency 3        174
        # currency 4         10

        this_currency = qLocal.CREDIT_CURRENCY.values[0]

        fConvertFactor = g_currency_converter[this_currency]

        # Drop currency. Could have been kept as a signature property, but there are very few non 'currency 1.'

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
        self._amt_credit_max_overdue = fConvertFactor * qLocal.AMT_CREDIT_MAX_OVERDUE.values[0]
        
        #       Credit amount for the Credit Bureau credit,
        self._amt_credit_sum = fConvertFactor * qLocal.AMT_CREDIT_SUM.values[0]

        #       Debt on Credit Bureau credit,
        self._amt_credit_sum_debt = fConvertFactor * qLocal.AMT_CREDIT_SUM_DEBT.values[0]

        #       Credit limit of credit card reported in Credit Bureau,
        self._amt_credit_sum_limit = fConvertFactor * qLocal.AMT_CREDIT_SUM_LIMIT.values[0]

        #       Amount overdue on Credit Bureau credit,
        self._amt_credit_sum_overdue = fConvertFactor * qLocal.AMT_CREDIT_SUM_OVERDUE.values[0]

        # Annuity of the Credit Bureau credit,
        self._amt_annuity = fConvertFactor * qLocal.AMT_ANNUITY.values[0]

    def GetCreditActiveDesc(self):
        desc = g_Cat["Bureau_CreditActive"][self._credit_active]
        return desc

    def GetCreditCurrencyDesc(self):
        desc = g_Cat["Bureau_CreditCurrency"][self._credit_currency]
        return desc

    def GetCreditTypeDesc(self):
        desc = g_Cat["Bureau_CreditType"][self._credit_type]
        return desc

    def Desc(self):
        print(self._dict_detail)
        
        print(f"Credit type: {self.GetCreditTypeDesc()}")
        print(f"Credit active: {self.GetCreditActiveDesc()}")
       
        print(f"Days credit: {self._days_credit}")
       
        print(f"Credit day overdue: {self._credit_day_overdue}")
        print(f"_days_credit_enddate: {self._days_credit_enddate}")
        print(f"_days_enddate_fact: {self._days_enddate_fact}")
        print(f"_days_credit_update: {self._days_credit_update}")
       
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

user_id = [370747]
q_b = get_on_sk_id_curr(user_id, 'b')


# For all bureau loans. Only first for now:
q = q_b[0:1]


get_user_data(370747)




"""c"""

def Init(qLocal):

    sk_id_curr = qLocal.SK_ID_CURR.values[0]

    b_id = qLocal.SK_ID_BUREAU.values[0]

    _dict_detail = detail_loan_bureau(b_id)
    _credit_active = qLocal.CREDIT_ACTIVE.cat.codes.values[0]
    _credit_currency = qLocal.CREDIT_CURRENCY.cat.codes.values[0]
    _days_credit = qLocal.DAYS_CREDIT.values[0]
    _credit_type = qLocal.CREDIT_TYPE.cat.codes.values[0]
    _credit_day_overdue = qLocal.CREDIT_DAY_OVERDUE.values[0]
    _days_credit_enddate = qLocal.DAYS_CREDIT_ENDDATE.values[0]
    _days_enddate_fact = qLocal.DAYS_ENDDATE_FACT.values[0]
    _amt_credit_max_overdue = qLocal.AMT_CREDIT_MAX_OVERDUE.values[0]
    _cnt_credit_prolong = qLocal.CNT_CREDIT_PROLONG.values[0]
    _amt_credit_sum = qLocal.AMT_CREDIT_SUM.values[0]
    _amt_credit_sum_debt = qLocal.AMT_CREDIT_SUM_DEBT.values[0]
    _amt_credit_sum_limit = qLocal.AMT_CREDIT_SUM_LIMIT.values[0]
    _amt_credit_sum_overdue = qLocal.AMT_CREDIT_SUM_OVERDUE.values[0]
    _days_credit_update = qLocal.DAYS_CREDIT_UPDATE.values[0]
    _amt_annuity = qLocal.AMT_ANNUITY.values[0]

"""c"""


