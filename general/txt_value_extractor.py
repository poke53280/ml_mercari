

import re
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=200)
np.core.arrayprint._line_width = 480






###################################################################################################
#
#    getSeriesFromListofLists
#
#

def getSeriesFromListofLists(ll, iCol):

    l_out = []

    for x in ll:
        l_out.append(x[iCol] if len(x) > iCol else 0)

    return l_out
"""c"""

###################################################################################################
#
#    extract_and_mask
#
#

def extract_and_mask(s, reg_ex, maskstr, num_cols):
    l_extract = []

    for x in s:
        l_extract.append(re.findall(reg_ex, x))
    
    l_removed = []

    for x in l:
        x = re.sub(reg_ex, maskstr, x)
        l_removed.append(x)

    d = {}

    d['s_masked'] = pd.Series(l_removed)

    for iCol in range(num_cols):
        n = pd.Series(getSeriesFromListofLists(l_extract, iCol))

        d[f"n{iCol}"] = n


    return pd.DataFrame(d)
"""c"""


l = ["Hello at 010349, hello 99321 where 994417", "Hello AAT9741.pop0000"]

sText = pd.Series(l)
df = pd.DataFrame(sText)

df_out = extract_and_mask(df["pim"], reg_exp_dato, "#DATO#", 3)









l_sakID = []
l_dato = []


for x in sText:
    l_dato.append(re.findall(reg_exp_dato, x))
    l_sakID.append(re.findall(reg_exp_saksbehandler, x))

"""c"""

l_removed = []

for x in l:
    x = re.sub(reg_exp_saksbehandler, "#SAKSB#", x)
    x = re.sub(reg_exp_dato, "#DATO#", x)

    l_removed.append(x)
"""c"""

sText_masked = pd.Series(l_removed)




d0 = pd.Series(getSeriesFromListofLists(l_dato, 0))
d1 = pd.Series(getSeriesFromListofLists(l_dato, 1))
d2 = pd.Series(getSeriesFromListofLists(l_dato, 2))

s0 = pd.Series(getSeriesFromListofLists(l_sakID, 0))
s1 = pd.Series(getSeriesFromListofLists(l_sakID, 1))
s2 = pd.Series(getSeriesFromListofLists(l_sakID, 2))



df = pd.DataFrame({'In_txt': sText, 'Out_txt':sText_masked, 'd0': d0, 'd1': d1, 'd2': d2, 's0': s0, 's1': s1, 's2': s2})



