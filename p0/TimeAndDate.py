

# See poke53280.DataHub.Main.EpochDate.cpp for detailed holiday handling.


import pandas as pd
import datetime
import numpy as np

###########################################################################
#
#   TimeAndDate_GetSecondsSinceEpochSeries
#

def TimeAndDate_GetSecondsSinceEpochSeries(s):
    s0 = (pd.to_datetime(s) - datetime.datetime(1970, 1, 1))
    s0 = s0/ np.timedelta64(1, 's')
    s0 = s0.astype(np.int64)

    return s0

"""c"""

###########################################################
#
#    TimeAndDate_GetDaysSinceEpoch
#
#

def TimeAndDate_GetDaysSinceEpoch(s, epoch, na_date):
    # From string to datetime, set NaT at error
    s_out = pd.to_datetime(s, errors = 'coerce')

    # Set NA to 2019, assume still active in job.
    s_out = s_out.fillna(future)

    # Offset from UNIX epoch
    s_out = s_out - epoch

    # Fast conversions to int.

    s_out = s_out.astype('timedelta64[D]')
    s_out = s_out.astype(int)

    return s_out



##########################################################################
#
# TimeAndDate_Get_random_epoch_birth_day
#

def TimeAndDate_Get_random_epoch_birth_day(fid):
    assert (len(fid) == 11)

    birth_year = int (fid[4:6])

    if birth_year > 20:
        birth_year = birth_year + 1900
    else:
        birth_year = birth_year + 2000

    start_date = datetime.date(day=1, month=1, year=birth_year).toordinal()

    end_date = datetime.date(day=31, month=12, year=birth_year).toordinal()

    random_day = datetime.date.fromordinal(random.randint(start_date, end_date))

    epoch_day = datetime.date(1970, 1, 1)

    days_since_epocy = (random_day - epoch_day).days 

    return days_since_epocy


##############################################################################
#
#   TimeAndDate_GetDateTimeFromEpochDays()
#
#

def TimeAndDate_GetDateTimeFromEpochDays(nEpochDays):
    return datetime.datetime(1970,1,1,0,0) + datetime.timedelta(nEpochDays)

"""c"""

def TimeAndDate_GetEpochDaysFromDateTime(d):
    return (d - datetime.datetime(1970,1,1,0,0)).days
"""c"""

##############################################################################
#
#   TimeAndDate_GetDateTimeDescription()
#
#

def TimeAndDate_GetDateTimeDescription(datetime):
    return datetime.strftime("%d.%m.%Y")

"""c"""


########################################################################
#
#   TimeAndDate_Get_birth_year_from_fid
#

def TimeAndDate_Get_birth_year_from_fid(fid):
    if len(fid) == 11:
        try:
            b = fid[4:6]
            return int (b)
        except ValueError:
            print(f"Warning bad fid: {fid}")
            return -1

    else:
        return -1


###############################################################################
#
#   TimeAndDate_AddNoise
#
#   Returns numpy array with value randomly changed +/- input number of days
#

def TimeAndDate_AddNoise(s, num_days):

    null_count = s.isnull().sum()
    assert (null_count == 0)

    sn = s.values
    noise = np.random.randint(-num_days, num_days +1, size = len(s))

    sn = sn + noise
    return sn

"""c"""

def calc_easter(year):
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

    return TimeAndDate_GetEpochDaysFromDateTime(d)


# Count xmas and new year, as full, half, not at all as holiday.
# Count wednesday before eastern as half or not at all as holiday.
#
# Include, seek to include vinterferie, høstferie.
#



# Returns epoch days off - week end and holidays. iYearEnd inclusive.
def get_days_off(iYearBegin, iYearEnd):

    l_holiday = []

    lYears = list (range (iYearBegin, iYearEnd + 1))

    for iYear in lYears:
    
        l_holiday.append(TimeAndDate_GetEpochDaysFromDateTime(datetime.datetime(iYear, 1, 1)))  # 1st Jan

        nEasterSunday = calc_easter(iYear)

        l_holiday.append(nEasterSunday - 3)  #Holy Thursday
        l_holiday.append(nEasterSunday - 2)  #Good Friday
        l_holiday.append(nEasterSunday + 1)  #2nd day easter

        nAscension = nEasterSunday + 39

        l_holiday.append(nAscension)

        nPenteCostSunday = nEasterSunday + 7 * 7

        l_holiday.append(nPenteCostSunday + 1)  #2nd day pentecost

        l_holiday.append(TimeAndDate_GetEpochDaysFromDateTime(datetime.datetime(iYear, 5, 1))) #1st of May

        l_holiday.append(TimeAndDate_GetEpochDaysFromDateTime(datetime.datetime(iYear, 5, 17))) #17th of May

        nXmas0 = TimeAndDate_GetEpochDaysFromDateTime(datetime.datetime(iYear, 12, 24))

        l_holiday.append(nXmas0)
        l_holiday.append(nXmas0 + 1)
        l_holiday.append(nXmas0 + 2)

        l_holiday.append(TimeAndDate_GetEpochDaysFromDateTime(datetime.datetime(iYear, 12, 31))) #new year celeb

    s_holiday = set(l_holiday)

    # All days in period:
    nFirst = TimeAndDate_GetEpochDaysFromDateTime(datetime.datetime(iYearBegin, 1, 1))   # Inclusive
    nLast = TimeAndDate_GetEpochDaysFromDateTime(datetime.datetime(iYearEnd, 12, 31))  # Inclusive

    anDays = np.array(range(nFirst,nLast + 1))

    # Find all Saturdays and Sundays

    nSaturdayBase = 2
    nSundayBase = 3   # Found out beforehand
    assert (TimeAndDate_GetDateTimeFromEpochDays(nSundayBase).weekday() == 6), "Sunday assertion"

    m_sunday = (anDays - nSundayBase) % 7 == 0
    m_saturday =  (anDays - nSaturdayBase) % 7 == 0

    anWeekendDays = anDays[m_saturday | m_sunday]

    s_weekend = set(anWeekendDays)

    s_off_total = s_holiday.union(s_weekend)

    an = np.array(list (s_off_total))

    an = np.sort(an)

    return an



def get_off_ratio(nDayBegin, nDayEnd, anOff):

    assert nDayEnd >= nDayBegin, "nDayEnd >= nDayBegin"

    anDays = np.array(range(nDayBegin, nDayEnd + 1))

    # Check that days off range is suffeciently large to cover input days:

    assert np.max(anOff) > np.max(anDays), "np.max(anOff) > np.max(anDays)"
    assert np.min(anOff) < np.min(anDays), "np.max(anOff) > np.max(anDays)"

    m = np.isin(anDays, anOff)

    nOff = (m == True).sum()
    nOn = (m == False).sum()

    rOffRatio = 1.0 * nOff / len (m)

    return rOffRatio

"""c"""

def create_off_feature(df):

    # Into numpy arrays for calculations:

    anFrom = np.array (df.F0)
    anTo = np.array (df.T0)

    anTotalX = np.empty(shape = len(anFrom), dtype = np.float)

    lRange = list (range (len(anFrom)))

    for i in lRange:
        r = get_off_ratio(anFrom[i], anTo[i], anOff)
        anTotalX[i] = r

    sOff = pd.Series(anTotalX)

    return sOff

"""c"""


# Usage

# FIX ASSERTION ON E.G 2007, 2008
anOff = get_days_off(1970, 2050)


an1 = np.array([8300, 8310, 8360, 8390])
an2   = np.array([8303, 8314, 8366, 8397])

df = pd.DataFrame({'F0': an1, 'T0': an2})

s = create_off_feature(df)

df = df.assign(rOff = s)

df

