

import pandas as pd
import datetime
import numpy as np


def _GetDateTimeFromEpochDays(nEpochDays):
    return datetime.datetime(1970,1,1,0,0) + datetime.timedelta(nEpochDays)

"""c"""

def _GetEpochDaysFromDateTime(d):
    return (d - datetime.datetime(1970,1,1,0,0)).days
"""c"""


def _calc_easter(year):
    "Returns Easter as a date object."
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    month = f // 31
    day = f % 31 + 1    

    d = datetime.datetime(year, month, day)

    return _GetEpochDaysFromDateTime(d)


# Count xmas and new year, as full, half, not at all as holiday.
# Count wednesday before eastern as half or not at all as holiday.
#
# Include, seek to include vinterferie, høstferie.
#



# Returns epoch days off - week end and holidays. iYearEnd inclusive.

def _get_days_off(iYearBegin, iYearEnd):

    l_holiday = []

    lYears = list (range (iYearBegin, iYearEnd + 1))

    for iYear in lYears:
    
        l_holiday.append(_GetEpochDaysFromDateTime(datetime.datetime(iYear, 1, 1)))  # 1st Jan

        nEasterSunday = _calc_easter(iYear)

        l_holiday.append(nEasterSunday - 3)  #Holy Thursday
        l_holiday.append(nEasterSunday - 2)  #Good Friday
        l_holiday.append(nEasterSunday + 1)  #2nd day easter

        nAscension = nEasterSunday + 39

        l_holiday.append(nAscension)

        nPenteCostSunday = nEasterSunday + 7 * 7

        l_holiday.append(nPenteCostSunday + 1)  #2nd day pentecost

        l_holiday.append(_GetEpochDaysFromDateTime(datetime.datetime(iYear, 5, 1))) #1st of May

        l_holiday.append(_GetEpochDaysFromDateTime(datetime.datetime(iYear, 5, 17))) #17th of May

        nXmas0 = _GetEpochDaysFromDateTime(datetime.datetime(iYear, 12, 24))

        l_holiday.append(nXmas0)
        l_holiday.append(nXmas0 + 1)
        l_holiday.append(nXmas0 + 2)

        l_holiday.append(_GetEpochDaysFromDateTime(datetime.datetime(iYear, 12, 31))) #new year celeb

    s_holiday = set(l_holiday)

    # All days in period:
    nFirst = _GetEpochDaysFromDateTime(datetime.datetime(iYearBegin, 1, 1))   # Inclusive
    nLast = _GetEpochDaysFromDateTime(datetime.datetime(iYearEnd, 12, 31))  # Inclusive

    anDays = np.array(range(nFirst,nLast + 1))

    # Find all Saturdays and Sundays

    nSaturdayBase = 2
    nSundayBase = 3   # Found out beforehand
    assert (_GetDateTimeFromEpochDays(nSundayBase).weekday() == 6), "Sunday assertion"

    m_sunday = (anDays - nSundayBase) % 7 == 0
    m_saturday =  (anDays - nSaturdayBase) % 7 == 0

    anWeekendDays = anDays[m_saturday | m_sunday]

    s_weekend = set(anWeekendDays)

    s_off_total = s_holiday.union(s_weekend)

    an = np.array(list (s_off_total))

    an = np.sort(an)

    return an



def _get_off_number(nDayBegin, nDayEnd, anOff):

    assert nDayEnd >= nDayBegin, "nDayEnd >= nDayBegin"

    anDays = np.array(range(nDayBegin, nDayEnd + 1))

    # Check that days off range is suffeciently large to cover input days:

    assert np.max(anOff) > np.max(anDays), "np.max(anOff) > np.max(anDays)"
    assert np.min(anOff) < np.min(anDays), "np.max(anOff) > np.max(anDays)"

    m = np.isin(anDays, anOff)

    nOff = (m == True).sum()
    nOn = (m == False).sum()

    return nOff

"""c"""

def GetHolidayFeatures(df, zFrom, zTo):

    anOff = _get_days_off(1970, 2050)

    # Into numpy arrays for calculations:

    anFrom = np.array (df[zFrom])
    anTo = np.array (df[zTo])

    anTotalX = np.empty(shape = len(anFrom), dtype = np.int)


    anOffConnectedStart = np.empty(shape = len(anFrom), dtype = np.bool)
    anOffConnectedEnd = np.empty(shape = len(anFrom), dtype = np.bool)

    lRange = list (range (len(anFrom)))

    for i in lRange:
        r = _get_off_number(anFrom[i], anTo[i], anOff)
        anTotalX[i] = r

        nDayBeforeFrom = anFrom[i] - 1
        nDayAfterTo = anTo[i] + 1

        anOffConnectedStart[i] = _get_off_number(nDayBeforeFrom, nDayBeforeFrom, anOff) > 0
        anOffConnectedEnd[i] = _get_off_number(nDayAfterTo, nDayAfterTo, anOff) > 0

    return pd.Series(anTotalX), pd.Series(anOffConnectedStart), pd.Series(anOffConnectedEnd)

"""c"""


def holiday_example():

    # Create a dataframe with to and from values given in days since epoch 1.jan 1970.

    an1 = np.array([8300, 8310, 8360, 8390])
    an2 = np.array([8303, 8314, 8366, 8397])

    df = pd.DataFrame({'F0': an1, 'T0': an2})


    # Bring in the named start and end column names as parameters

    hL, hS, hE = GetHolidayFeatures(df, 'F0', 'T0')

    # Returned are
    # hL: Number of 'off days': Number of week end days + Norwegian holiday days.
    # Christmas (24th Dec), Last day of year (31th Dec) are currenty interpreted as holidays.

    # hS: True if the day before the from day is an off day, i.e. if the period is connected to an off day at the start.
    # hE: True if the day after the to day is an off day, i.e. if the period is connected to an off day at the end side.

    # Verify results by calling _GetDateTimeFromEpochDays(8300) et.c. and check with a calendar.

    df = df.assign(hL = hL, hS = hS, hE = hE)

    print (df)

    #        F0    T0  hL     hS     hE
    #   0  8300  8303   0  False   True
    #   1  8310  8314   2  False  False
    #   2  8360  8366   2  False   True
    #   3  8390  8397   4   True  False

"""c"""










