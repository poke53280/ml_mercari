

import re
import numpy as np
import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=200)
np.core.arrayprint._line_width = 480


# Parse dates for validity


def validate_long_date(sDate):
    isValidated = True

    if not isinstance(sDate, str):
        return False

    try:
        a = datetime.strptime(sDate, "%d%m%y")
    except ValueError:
        isValidated = False

    return isValidated
"""c"""

def validate_short_date(sDate, sFullYear):
    sLongDate = sDate + sFullYear[2:]
    return validate_long_date(sLongDate)
"""c"""



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

def SetConsecutive(keyword, replacement, l_removed):

    l_consecutive = []

    for x in l_removed:

        iCount = 0
        while True:
            x_new = x.replace(keyword, f"{replacement}{iCount}_", 1)
            isUnchanged = (x_new == x)

            x = x_new

            if isUnchanged:
                break

            iCount = iCount + 1
        l_consecutive.append(x)

    return l_consecutive
"""c"""


l = ["Hello at 010349, hello 99321 where 994417", "Hello AAT9741.pop0000 from 0102-020215"]

sText = pd.Series(l)

df = pd.DataFrame({'Tekst': sText})


# Better: Use datetime.

reg_exp_dato_long = re.compile(' \d{6}?!\d)')
reg_exp_dato_short= re.compile(r' \d{4}(?!\d)')
reg_exp_saksbehandler = re.compile('[A-Za-z]{3}\d{4}')

reg_exp_fid = re.compile(' \d{11}?!\d)')



l_sakID = []
l_dato_long = []
l_dato_short = []


for x in sText:
    l_dato_long.append(re.findall(reg_exp_dato_long, x))
    l_dato_short.append(re.findall(reg_exp_dato_short, x))
    l_sakID.append(re.findall(reg_exp_saksbehandler, x))

"""c"""

l_removed = []

for x in l:
    x = re.sub(reg_exp_saksbehandler, "#SAKSB#", x)
    x = re.sub(reg_exp_dato_long, " #LONG_DATE#", x)
    x = re.sub(reg_exp_dato_short, " #SHORT_DATE#", x)
    x = re.sub(reg_exp_fid, " #FID#", x)

    l_removed.append(x)
"""c"""


l_parsed = SetConsecutive("#SAKSB#", "SB", l_removed)
l_parsed = SetConsecutive("#LONG_DATE#", "L", l_parsed)
l_parsed = SetConsecutive("#SHORT_DATE#", "S", l_parsed)


sText_masked = pd.Series(l_parsed)

d0 = pd.Series(getSeriesFromListofLists(l_dato_long, 0))
d1 = pd.Series(getSeriesFromListofLists(l_dato_long, 1))
d2 = pd.Series(getSeriesFromListofLists(l_dato_long, 2))

s0 = pd.Series(getSeriesFromListofLists(l_dato_short, 0))
s1 = pd.Series(getSeriesFromListofLists(l_dato_short, 1))
s2 = pd.Series(getSeriesFromListofLists(l_dato_short, 2))

sak0 = pd.Series(getSeriesFromListofLists(l_sakID, 0))
sak1 = pd.Series(getSeriesFromListofLists(l_sakID, 1))
sak2 = pd.Series(getSeriesFromListofLists(l_sakID, 2))

df = pd.DataFrame({'In_txt': sText, 'Out_txt':sText_masked, 'd0': d0, 'd1': d1, 'd2': d2, 's0':s0, 's1':s1, 's2': s2, 'sak0': sak0, 'sak1': sak1, 'sak2': sak2})




