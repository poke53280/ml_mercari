
import os
import gc
import pandas as pd
import hashlib


local_dir = os.getenv('LOCAL_PY_DIR')
assert local_dir is not None, "Set environment variable LOCAL_DIR to location of this python file."
print(f"Local python directoy is set to {local_dir}")
os.chdir(local_dir)


import ai_lab_datapipe.DataProvider
from ai_lab_datapipe.DataGeneral import toDaysSinceEpoch


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def left_merge(df, add_df, left_id, right_id):

    col_left = df.columns
    col_right = add_df.columns

    assert left_id in col_left, f"'{left_id}' not a column in left data frame"
    assert right_id in col_right, f"'{right_id}' not a column in right data frame"

    assert len(set(col_left).intersection(set(col_right))) == 0, "Identrical column names in left and right => Rename"
    
    q = df.merge(add_df, how = 'left', left_on=left_id, right_on = right_id)

    # Check NA
    nNAs = q[right_id].isnull().sum()

    df = q

    # DROP IDs 
    df = df.drop([left_id, right_id], axis = 1)

    if nNAs == 0:
        print("Merge: Found match for all rows")
    else:
        print(f"NAs for {nNAs} row(s)")

    return df        
"""c"""

def apply_hash(x):
    hash_object = hashlib.md5(x.encode())
    return hash_object.hexdigest()


def get_SF(d):

    CONFIG_DATA_START = "2006-01-01"

    config_file = os.getenv('DB_DATA')
    assert config_file is not None, "No config file found in environment variable DB_DATA"
    print(f"Using database connection file '{config_file}'")


    os.environ['NLS_NCHAR_CHARACTERSET']='AL16UTF16'
    os.environ['NLS_CHARACTERSET']='WE8ISO8859P15'
    os.environ['NLS_LANG']='AMERICAN_AMERICA.WE8ISO8859P15'


    f = open(config_file,"r")
    config = f.read()
    f.close()

    dp = ai_lab_datapipe.DataProvider.DataProvider(config)

    l_queries = []

    l_queries.append( ("A", "DIM_INNTEKT",     "dim_inntekt") )
    l_queries.append( ("A", "DIM_SYKMELDER",   "dim_sykmelder") )
    l_queries.append( ("A", "DIM_NAERING",   "dim_naering") )
    l_queries.append( ("A", "DIM_DIAGNOSE",   "dim_diagnose") )
    l_queries.append( ("A", "DIM_GEOGRAFI",   "dim_geografi") )
    l_queries.append( ("A", "DIM_LAND",   "dim_land") )
    l_queries.append( ("A", "DIM_TID",   "dim_tid") )
    l_queries.append( ("A", "DIM_VARIGHET",   "dim_varighet") )
    l_queries.append( ("A", "DIM_VIRKSOMHET",   "dim_virksomhet") )
    l_queries.append( ("A", "DIM_YRKE",   "dim_yrke") )
    l_queries.append( ("A", "NAV_UNIT",   "navunit") )

    d.update( dp.async_load(l_queries))

    d_s = dp.async_load([("A", "SYKMELDING", "melding")])
    d_p = dp.async_load([("A", "DIM_PERSON", "dim_person")])
    d_f = dp.async_load([("A", "SSB_FRAVAR", "fravar") ])

    d['melding']  = d_s['melding']
    d['dim_person'] = d_p['dim_person']
    d['fravar'] = d_f['fravar']

    del d_s
    del d_p
    del d_f

    gc.collect()

    print(f"Number of dataframes: {len(d)}")

    d['melding'].columns = ['id', 'income_id', 'md_id', 'naering_id', 'unit_id', 'diag_id', 'F0', 'F1', 'T1']
    d['dim_inntekt'].columns = ['income_id', 'income_p_cat', 'income_cat']
    d['dim_diagnose'].columns = ['diag_id', 'diag_code', 'diag_sub1_code', 'diag_sub2_code', 'diag_icpc2_kode', 'diag_group_code', 'diag_icd_g1_code', 'diag_icd_g2_code', 'diag_smp_main_code', 'diag_smp_sub_code', 'diag_group_cat']
    d['dim_naering'].columns = ['business_type_id', 'business_type_code']
    d['navunit'].columns =  ['unit_id', 'unit_code', 'unit_l3_code', 'unit_l2_code', 'unit_l1_code', 'unit_field_code', 'unit_info_source']
    d['dim_sykmelder'].columns = ['md_id', 'md_year', 'md_birthFORHASHONLT', 'md_nameFORHASHONLY', 'md_gender', 'md_code', 'md_type_code', 'md_g1_code', 'md_main_flag', 'md_sub_code', 'md_sub_flag', 'md_dept_code', 'md_field_code', 'md_cmpr_type_code', 'md_from', 'md_to']
    d['dim_person'].columns = ['dim_id', 'id', 'valid_from_date', 'valid_to_date', 'valid_flag', 'gender', 'country_born', 'country_citizenship', 'country_living', 'country_address', 'dim_geo_living', 'dim_geo_living_last', 'dim_geo_address', 'born_year', 'state_code', 'state_date']
    d['dim_geografi'].columns = ['geo_id', 'geo_country_code', 'geo_county', 'geo_commune', 'geo_town_nr', 'geo_postal_code']
    d['dim_land'].columns = ['country_id', 'country_code', 'country_nordic', 'country_eu', 'country_oecd', 'country_eos', 'country_western', 'country_continent']
    d['fravar'].columns = ['dim_id', 'ssb_dim_time', 'ssb_dim_duration', 'ssb_dim_virksomhet', 'ssb_dim_naering', 'ssb_dim_yrke', 'ssb_dim_geo_workplace', 'ssb_lost_days', 'ssb_leave_mean_rate_pct']
    d['dim_tid'].columns = ['time_id', 'date', 'week_day']
    d['dim_varighet'].columns = ['duration_id', 'duration_days']
    d['dim_virksomhet'].columns = ['company_id', 'company_num_employees', 'companY_business_area_code', 'company_type', 'company_postal_code', 'company_dim_geografi_county', 'enterprise_num_employees', 'enterprise_stock_value', 'enterprise_type', 'enterprise_postal_code', 'enterprise_sector_code', 'enterprise_ia_start_date', 'org_num_employees', 'company_valid_from_date', 'company_valid_to_date', 'company_active_flag']
    d['dim_yrke'].columns = ['prof_id', 'prof_nav_code', 'prof_styrk_code']


    # -----------------------------------------------------------------------------------------------
    # PROCESS BEGIN dim_sykmelder
    #
    # Create unique ID for sykmelder, then hash and categorize.
    q = d['dim_sykmelder']

    s1 = q['md_birthFORHASHONLT'].astype('str')
    s = q['md_nameFORHASHONLY'] + s1

    s_hash = s.apply(apply_hash)
    
    d['dim_sykmelder'] = d['dim_sykmelder'].assign(PID_HASH = s_hash)

    q = d['dim_sykmelder']['PID_HASH'].astype('category').cat.codes

    print(f"Num sykmelders= {len(q)}, unique name/birth = {q.nunique()}")

    d['dim_sykmelder'] = d['dim_sykmelder'].assign(PID_CAT = q)

    d['dim_sykmelder'] = d['dim_sykmelder'].drop(['md_birthFORHASHONLT', 'md_nameFORHASHONLY', 'PID_HASH'], axis = 1)

    # PROCESS END DIM_SYKMELDER


    # Virksomhet keep hash ID. Store a copy in column, since left_merge deletes the IDs.

    d['dim_virksomhet']['C_ID'] = d['dim_virksomhet']['company_id'].astype('category').cat.codes

    # Person keep hash id.
    d['dim_person']['FK_ID'] = d['dim_person']['id'].astype('category').cat.codes


    # Merge in date on fravar to be able to cut early data (< 01.01.2006).

    d['fravar'] = left_merge(d['fravar'], d['dim_tid'], 'ssb_dim_time', 'time_id')

    from_time = toDaysSinceEpoch(CONFIG_DATA_START)

    m = d['fravar'].date < 1.0 * from_time

    d['fravar'] = d['fravar'][~m]

    len (d['fravar'])

    
    s_p_id_dim = set(d['dim_person'].dim_id)

    m_fravar_dim_id_in_person = d['fravar'].dim_id.isin(s_p_id_dim)

    assert len(d['fravar']) == m_fravar_dim_id_in_person.sum(), "Persons in fravar not found in dim_person"

    # Find all FK_PERSON1 for fravar
    q = d['dim_person'][ ['dim_id', 'id']]

    q.columns = ['dim_idxxx', 'idxxx']


    d['fravar']['dim_id_backup'] = d['fravar']['dim_id']

    qf = left_merge(d['fravar'], q, 'dim_id', 'dim_idxxx')

    # FK_PERSON in fravar:
    s_fk_person_fravar = set(qf.idxxx)

    # FK_PERSON in melding:
    s_fk_person_melding = set(d['melding'].id)

    s_fk_person_all = s_fk_person_fravar.union(s_fk_person_melding)

    m = d['dim_person'].id.isin(s_fk_person_all)

    d['dim_person'] = d['dim_person'][m]

    # Now only relevant persons (as per DPIA) in memory.





d = {}

get_SF(d)




# Check point. All db access done. All out of project range data cut away.

# Merge duration for fravar

d['fravar'] = left_merge(d['fravar'], d['dim_varighet'], 'ssb_dim_duration', 'duration_id')
d['fravar'] = left_merge(d['fravar'], d['dim_yrke'], 'ssb_dim_yrke', 'prof_id')
d['fravar'] = left_merge(d['fravar'], d['dim_virksomhet'], 'ssb_dim_virksomhet', 'company_id')
d['fravar'] = left_merge(d['fravar'], d['dim_naering'], 'ssb_dim_naering', 'business_type_id')


# Delete tables done with

d['fravar']

del d['dim_tid']
del d['dim_virksomhet']
del d['dim_varighet']
del d['dim_yrke']




d['fravar'] = left_merge(d['fravar'], d['dim_geografi'],'ssb_dim_geo_workplace', 'geo_id')


cols = d['fravar'].columns

new_cols = []

for c in cols:
    if 'geo' in c:
        new_cols.append('workplace_' + c)
    else:
        new_cols.append(c)


d['fravar'].columns = new_cols


new_cols[0] = 'XX_dim_id'
d['fravar'].columns = new_cols


d['fravar'] = left_merge(d['fravar'], d['dim_person'], 'XX_dim_id', 'dim_id')

# Todo - return match factor (or similar)


d['fravar'] = left_merge(d['fravar'], d['dim_geografi'],'dim_geo_living', 'geo_id')


cols = d['fravar'].columns

new_cols = []

for c in cols:
    if 'geo' in c:
        new_cols.append('living_' + c)
    else:
        new_cols.append(c)


d['fravar'].columns = new_cols

# FIX _ Adds 'living' to 'workplace' as well. et.c.

# CHECK - no postal codes?




##################### tests


# one id:

l_id = [730412047, 2520904400, 2520976142, 730375555, 730359310]

id = l_id[4]


print(f"Retrieve information on id = {id}")


m = d['melding'].id == id

q_melding = d['melding'][m].copy()

print(f"Found {len(q_melding)} leave record(s)")

m = d['dim_person'].id == id

q_person = d['dim_person'][m]

print(f"Found {len(q_person)} person record(s) on id")

l_dim_id = list (q_person.dim_id)

m = d['fravar'].dim_id.isin(l_dim_id)

q_fravar = d['fravar'][m]

print(f"Found {len(q_fravar)} ssb leave record(s)")


q_melding
q_fravar




df = d['syk'].copy()

def count_invalid(df, s):
    m = df[s] < 0
    return m.sum()

count_invalid(df, 'diag_id')

df = df.merge(d['age'], how = 'left', left_on = 'age_id' , right_on = 'age_id')
df = df.merge(d['income'], how = 'left', left_on = 'income_id' , right_on = 'income_id')
df = df.merge(d['diag'], how = 'left', left_on = 'diag_id' , right_on = 'diag_id')
df = df.merge(d['work'], how = 'left', on = 'work_id')
df = df.merge(d['unit'], how = 'left', on = 'unit_id')
df = df.merge(d['md'], how = 'left', on = 'md_id')

df = df.drop(['age_id', 'income_id', 'md_id', 'work_id', 'unit_id', 'diag_id'], axis = 1)

df = df.drop(['income_p_cat'], axis = 1)  # No variance



df = df.sort_values(by = ['id', 'F0', 'F1'])

# SM approach completed.


# SF approach
l_queries = []
l_queries.append(("A", "SF_Fravar",            "ssb_sf") )

e = dp.async_load(l_queries)

# min max dato
de = e['ssb_sf']

#
#...XXX Continue here
#


def old_db_out(df):

    l_queries = []

    l_queries.append( ("A", "sql_syk3",      "syk") )
    #l_queries.append( ("A", "sql_fravar",    "fravar") )
    l_queries.append( ("A", "sql_pmap",      "pmap") )
    #l_queries.append( ("B", "sql_vedtak",    "vedtak") )
    #l_queries.append( ("B", "sql_meldekort", "meldekort") )
    #l_queries.append( ("C", "select_large",  "aa") )

    d = dp.async_load(l_queries)

    p = d['pmap'].copy()


    p = d['pmap'].copy()

    p.columns = ['FID', 'FK', 'A']

    m = p.FK == -1

    p = p[~m]

    # Clean up FID to FK mapping.

    p_state = p["FID"].apply(classifyFID)

    m = p_state == 'E'

    print(f"Removing {len(p[m])} rows with bad fid")

    p = p[~m]

    p_state = p["FID"].apply(classifyFID)
    p_epoch = p["FID"].apply(toDaysSinceEpochFromFID)

    p = p.assign(S = p_state)
    p = p.assign(E = p_epoch)

    # Prepare conversion dictionaries

    l_fid = p.FID.tolist()
    l_fk  = p.FK.tolist()
    l_a   = p.A.tolist()
    l_s   = p.S.tolist()
    l_e   = p.E.tolist()

    d_FK_TO_FID = dict (zip (l_fk, l_fid))
    d_FK_TO_E = dict (zip (l_fk, l_e))
    d_FK_TO_S = dict (zip (l_fk, l_s))


########################################################################################################


df_syk = d['syk'].copy()

df_syk.columns = ["FK", "DID", "F0", "F1", "T0", "D"]

# Data stream start time as documented in project DPIA. Discard earlier data.

syk_data_begin = toDaysSinceEpoch(CONFIG_DATA_START)

m = (df_syk.F0 >= syk_data_begin)

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data before {CONFIG_DATA_START}. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]

syk_data_future = toDaysSinceEpoch(CONFIG_DATA_INVALID_FUTURE)

m = (df_syk.F0 < syk_data_future) & (df_syk.F1 < syk_data_future)

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data after {CONFIG_DATA_INVALID_FUTURE}, assumed bad. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]

def isExists(x, d):
    return x in d

m = df_syk.FK.apply(isExists, args = (d_FK_TO_FID,))

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data where no FID <-> FK map found. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]

def getValue(x, d):
    return d[x]


fid = df_syk.FK.apply(getValue, args = (d_FK_TO_FID,))
birth = df_syk.FK.apply(getValue, args = (d_FK_TO_E,))
fid_s = df_syk.FK.apply(getValue, args = (d_FK_TO_S,))

df_syk = df_syk.assign(B = birth)
df_syk = df_syk.assign(FID = fid)
df_syk = df_syk.assign(FID_S = fid_s)

g = df_syk["FID"].apply(get_gender_from_fid)

df_syk = df_syk.assign(G = g)

q = df_syk.FID.astype('category').cat.codes

df_syk = df_syk.assign (IDX = q)

q = df_syk.DID.astype('category').cat.codes

df_syk = df_syk.assign (MD = q)

# drop later to verify deduced data. df_syk = df_syk.drop(["FK", "FID", "DID"], axis = 1)

df_syk = df_syk.assign(D = df_syk.D.astype('category'))
df_syk = df_syk.assign(FID_S = df_syk.FID_S.astype('category'))


# For brevity
df = df_syk

# Offset all dates into positive numbers for convenience.

first_day = np.min(  [df.F0.min(), df.F1.min(), df.T0.min(), df.B.min()])

df.F0 -= first_day
df.F1 -= first_day
df.T0 -= first_day
df.B -= first_day


min_day = np.min(  [df.F0.min(), df.F1.min(), df.T0.min(), df.B.min()])

assert min_day == 0

max_day = np.max(  [df.F0.max(), df.F1.max(), df.T0.max(), df.B.max()])

print(f"Max day after offset {first_day} : {max_day}")


def downcast_unsigned(s):
    assert s.min() >= 0
    return pd.to_numeric(s, downcast = 'unsigned')


# Convert data types to compact forms

df = df.assign(MD = downcast_unsigned(df.MD))
df = df.assign(IDX = downcast_unsigned(df.IDX))

df = df.assign(F0 = downcast_unsigned(df.F0))
df = df.assign(F1 = downcast_unsigned(df.F1))
df = df.assign(T0 = downcast_unsigned(df.T0))
df = df.assign(B = downcast_unsigned(df.B))

q = df_syk.FID_S.astype('category').cat.codes

df = df.assign(FID_S = q)
df = df.assign(FID_S = downcast_unsigned(df.FID_S))





df = df.sort_values(by = 'IDX')

df = df.reset_index(drop = True)

# Diagnose FE
q = df.D.apply(len)

df = df.assign(D_L = downcast_unsigned(q))

q = df.D.apply(lambda x: x[0])


# First letter
df = df.assign(D_H = q)

q = df.D.apply(lambda x: x[1:])

df = df.assign(D_C = q)

df = df.assign(D_H = df.D_H.astype('category'))
df = df.assign(D_C = df.D_C.astype('category'))

q = df.D_H.cat.codes
df = df.assign(D_H = q)

q = df.D_C.cat.codes
df = df.assign(D_C = q)

q = df.D.cat.codes
df = df.assign(D = q)

df = df.assign(D = downcast_unsigned(df.D))
df = df.assign(G = df.G.astype('uint8'))

df = df.assign(IDX =  downcast_unsigned(df.IDX))

df = df.assign(D_L = downcast_unsigned(df.D_L))

df = df.assign(D_H = downcast_unsigned(df.D_H))

df = df.assign(D_C = downcast_unsigned(df.D_C))

# No value in D now
# DROP LATERdf = df.drop(['D'], axis = 1)

df = df.sort_values(by = ['IDX', 'F0', 'F1', 'T0'])

df = df.reset_index(drop=True)


df = df.drop(['FK', 'DID', 'FID'], axis = 1)


# Main continutes below functions

from TimeLineTool import TimeLineText

def group_intervals(pt, nGrow):


    lf1 = []
    lq = []

    lf1 = pt.F1.values
    lq = pt.T0.values

    r_m = np.array((lf1,lq)).T

    nEdgeAir = 2 * nGrow + 5

    # To ensure all intervals are within accepted range
    t_start = lf1.min() - nEdgeAir
    t_end = lq.max() + nEdgeAir

    timelineText = TimeLineText(t_start, t_end, True, False, False, True)

    r_m_excl = r_m.copy()
    r_m_excl[:, 1] += 1

    r_m_processed = timelineText.CombineIntervals(r_m_excl, nGrow)

    # Back to inclusive mode
    r_m_processed[:, 1] -= 1

    assert len(r_m_processed) > 0, "No resulting groups"

    group_idx = np.zeros(r_m.shape[0])

    group_idx[:] = -1

    r_m_start = r_m[:, 0]
    r_m_end   = r_m[:, 1]

    # Place all intervals into the groups

    for idx, p in enumerate(r_m_processed):
        a = p[0]
        b = p[1]

        # Fully inside range:
        m = (r_m_start >= a) & (r_m_end <= b)

        # Inside range a, b:
        # nInside = len(r_m[m])

        group_idx[m] = idx

        # print(f"#intervals in range [{a}, {b}]: {nInside}")

    assert (group_idx < 0).sum() == 0, "Input interval(s) not assigned to any group"
    assert len(np.unique(group_idx)) == len(r_m_processed), "Found empty groups"

    return r_m_processed, group_idx

"""c"""


# Returns all data in focus as indices into input array r_m_processed. Last element is a valid target interval.
# May return empty index array when no valid target interval is found.

def getValidDataIntervals(r_m_processed, L_min):

    # Assert sorted in ascending order
    begin = r_m_processed[:, 0]
    end = r_m_processed[:, 1]

    begin_s = np.sort(begin)
    end_s = np.sort(end)

    assert (begin != begin_s).sum() == 0
    assert (end != end_s).sum() == 0

    # Filter out too early and too late

    nEarly = 32000  # -1 to disable early filter.
    nLate = -1   # -1 to disable late filter.

    lEarly = []
    lLate = []

    lInterest = []

    for idx, p in enumerate(r_m_processed):
        a = p[0]
        b = p[1]

        if nEarly > 0 and b < nEarly:
            # print(f"Early: {idx}")
            lEarly.append(idx)

        elif nLate > 0 and a > nLate:
            #print(f"Late: {idx}")
            lLate.append(idx)

        else:
            lInterest.append(idx)


    if len(lInterest) == 0:
        pass
        #print("No data in interest zone")
    
    idx_target_candidate = -1

    for idx in lInterest:

        p = r_m_processed[idx]

        target_begin, target_end, target_stitch = p[0], p[1], p[2]

        L_full = 1 + target_end - target_begin
        L_adj  = L_full - target_stitch

        if L_adj >= L_min:
            idx_target_candidate = idx

    """c"""

    m = np.array(lInterest) <= idx_target_candidate  # None smaller for no target found => empty array output

    p_data_idx = np.array(lInterest)[m]

    return p_data_idx

"""c"""


def full_analysis(q, nGrow, l_target_min, t_c):

    assert len (q) > 0
    assert np.unique(q.IDX).shape[0] == 1, "Multiple IDs input"

    n_cut_into_leave = l_target_min - 1 # All data until yesterday with respect to prediction time l_target_min

    TIME_0 = time.time()

    r_m_processed, group_idx = group_intervals(q, nGrow)

    TIME_1 = time.time()

    t_c['d1'] += (TIME_1 - TIME_0)


    ids_valid_intervals = getValidDataIntervals(r_m_processed, l_target_min)

    TIME_2 = time.time()
    t_c['d2'] += (TIME_2 - TIME_1)


    s = set (range(len(r_m_processed)))
    s -= set(ids_valid_intervals)

    ids_discarded_intervals = np.array(list(s))  # Out of range or in future to target interval

    assert ids_valid_intervals.shape[0] + ids_discarded_intervals.shape[0] == r_m_processed.shape[0], "ids_valid_intervals.shape[0] + ids_discarded_intervals.shape[0] == r_m_processed.shape[0]"

    isNoData = ids_valid_intervals.shape[0] == 0

    if isNoData:
        # print(f"No target found for {q.IDX.values[0]}")
        return (-1, -1, -1, None, None, t_c)

    
    ids_feature = ids_valid_intervals[:-1]      # Can be empty: Valid, common condition.

    idx_target = ids_valid_intervals[-1]

    m_discarded = np.isin(group_idx, ids_discarded_intervals)


    TIME_3 = time.time()
    t_c['d3'] += (TIME_3 - TIME_2)

    # Method end product:
    m_feature = np.isin(group_idx, ids_feature)

    targetData = r_m_processed[idx_target]

    target_begin, target_end, target_stitch = targetData

    L_full = 1 + target_end - target_begin
    L_adj  = L_full - target_stitch

    assert L_adj >= l_target_min, "Interval length < configured min target length"

    target_cut_day = target_begin + n_cut_into_leave

    m_valid_train_range = np.array(q.F1 <= target_cut_day)

    m_target = np.isin(group_idx, idx_target)

    TIME_4 = time.time()
    t_c['d4'] += (TIME_4 - TIME_3)

    # All elements: Discarded, in feature or in target

    anCount = np.zeros(q.shape[0])

    anCount[m_target] += 1
    anCount[m_feature] += 1
    anCount[m_discarded] += 1

    assert np.min(anCount) == 1 and np.max(anCount) == 1

    m_valid_train_target = m_target & m_valid_train_range

    # Data for target leave interval available CONF_CUT_INTO_LEAVE days(s) into leave.

    TIME_5 = time.time()
    t_c['d5'] += (TIME_5 - TIME_4)
   

    return (target_begin, L_full, L_adj, m_feature, m_valid_train_target, t_c)

"""c"""

N = 0  #0 for all

t_c = {'d_outer0': 0, 'd_outer1': 0, 'd1': 0, 'd2':0, 'd3':0, 'd4': 0, 'd5':0}

n = np.array(df.IDX.values).astype('int32')

nUnique = len (np.unique(n))

if N == 0:
    N = nUnique


assert len (np.unique(n)) >= N

print(f"Generating {N}/ {nUnique} data lines...")


l_IDX = list (range(N))

start_idx = 0
start_time = time.time()

l_Begin = []
l_LFull    = []
l_LAdj    = []


s = np.zeros(df.shape[0], dtype = int)

s[:] = 3

for x in l_IDX:

    if x > 0 and x % 10000 == 0:
        print(f"Analyzing idx = {x}/{N}...")

    T0 = time.time()

    end_idx = np.searchsorted(n, x+1)

    T1 = time.time()

    q = df[start_idx:end_idx]

    T2 = time.time()

    t_c['d_outer0'] += (T1 - T0)
    t_c['d_outer1'] += (T2 - T1)

    # print(f"start_idx = {start_idx}, end_idx = {end_idx}")

    begin, L_full, L_adj, m_f, m_t, t_c = full_analysis(q, 3, 7*7, t_c)

    if L_full < 0 or L_adj < 0:
        l_Begin.append(np.nan)
        l_LFull.append(np.nan)
        l_LAdj.append(np.nan)

    else:
        l_Begin.append(begin)
        l_LFull.append(L_full)
        l_LAdj.append(L_adj)

        v = s[start_idx:end_idx]

        v[:] = 2

        v[m_f] = 0
        v[m_t] = 1

    start_idx = end_idx


end_time = time.time()

d_time = end_time - start_time

print(f"t = {d_time}s. {N/d_time:.1f} items/s.")



df = df.assign(S = s)

df_target = pd.DataFrame({'ID': l_IDX, 'Begin': l_Begin,'Y_F': l_LFull, 'Y_Adj': l_LAdj })


m = df_target.Begin.isna()
nTrainIDs = df_target[~m].shape[0]
nAllIDs = df_target.shape[0]
rPCTTrain = 100.0 * nTrainIDs/ nAllIDs

print(f"{rPCTTrain:.1f}% of users provide training data")


#
# S info
# 0 : Part of feature
# 1 : Part of target
# 2 : Analyzed and discarded as too old or future data. Must be discarded from training set.
# 3 : Not analyzed


# Todo: Serialize 'df_raw'?


# Todo: 



Q_BACK = df.copy()
Q_BACK_TARGET = df_target.copy()

m = df.S < 2

df = df[m]   # Remove invalid and future data, and unanalyzed data.


df = df.assign(S = downcast_unsigned(df.S))


m = df_target.Begin.isna()

df_target = df_target[~m]

s = df_target.ID

s = s.reset_index(drop = True)

d = s.to_dict()

inv_map = {v: k for k, v in d.items()}

def newID(x, map):
    return map[x]

s = df.IDX.apply(newID, args = (inv_map,))

df = df.assign(IDX = s)

s = df_target.ID.apply(newID, args = (inv_map,))

df_target = df_target.assign(ID = s)


df_target = df_target.assign(Begin = downcast_unsigned(df_target.Begin))

s = df_target.Y_F - df_target.Y_Adj

df_target = df_target.assign(Y_S = downcast_unsigned(s))

df_target = df_target.drop(['Y_Adj'], axis = 1)

df_target = df_target.assign(Y_F = downcast_unsigned(df_target.Y_F))

df_target = df_target.assign(ID = downcast_unsigned(df_target.ID))



# Move FID_S, BIRTH and G from df to df_target

idx_to_fid_s = {}
idx_to_g = {}
idx_to_b = {}


def get_dicts(x):
    idx = x['IDX']
    fid_s = x['FID_S']
    b = x['B']
    g = x['G']

    idx_to_fid_s[idx] = fid_s
    idx_to_g[idx] = g
    idx_to_b[idx] = b


# Slow - 10 minutes?
df.apply(get_dicts, axis = 1)

assert len (idx_to_fid_s) == len (idx_to_g)
assert len (idx_to_fid_s) == len (idx_to_b)

def set_value(x, d):
    return d[x]

b = df_target.ID.apply(set_value, args = (idx_to_b, ))
g = df_target.ID.apply(set_value, args = (idx_to_g, ))
fid_s = df_target.ID.apply(set_value, args = (idx_to_fid_s, ))

df_target = df_target.assign(B = b)
df_target = df_target.assign(G = g)
df_target = df_target.assign(FS = fid_s)

df = df.drop(['B', 'G', 'FID_S'], axis = 1)

df.dtypes
df_target.dtypes

# Compacting types again

df_target = df_target.assign(B = downcast_unsigned(df_target.B))
df_target = df_target.assign(G = downcast_unsigned(df_target.G))
df_target = df_target.assign(FS = downcast_unsigned(df_target.FS))

df = df.assign(IDX = downcast_unsigned(df.IDX))

# Save point
now = datetime.datetime.now()

# Todo: Create sub folder and cd..  import os os.mkdir('Hello') 

df_target.to_pickle(DATA_DIR + now.strftime("df_t_%Y-%m-%d_%H-%M") + ".pkl")
df.to_pickle(DATA_DIR + now.strftime("df_%Y-%m-%d_%H-%M") + ".pkl")

#
# Todo: Emit description from template.
# Todo: Config and run clarified.
#
# 49 days
#  3 days grow. Policy explained.
# First day of dataset
# Discard day for feature set.
# Offset explained.
#
# Based on what code amd data. sql_version_string
#


