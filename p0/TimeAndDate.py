

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

