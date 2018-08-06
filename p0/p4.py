

###############################################################################
#
#       Analysis - timings with TimeLineTool.py
#
#

import pandas as pd
import numpy as np
import more_itertools as mit

import sys
import os

import numpy as np

sys.path

cwd = os.getcwd()

os.chdir('C:\\Users\\T149900\\ml_mercari')

from general.TimeLineTool import *


DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


df = pd.read_pickle(DATA_DIR + "noised_30JUL2018_cleaned.pkl")

df.shape[0]
df = df.drop_duplicates()
df.shape[0]



# Prepare dataframe for TimeLineTool.

df_m = pd.DataFrame()

df_m['IDX'] = df.ID  #Individual ID.
df_m['F1'] =  df.F1  #From day inclusive
df_m['Q'] =   df.T0  #To day inclusive

m = (df_m.Q < df_m.F1)

assert len (df[m]) == 0, "Invalid intervals given"

# t_start, t_end defined a time line interval:

t_start, t_end  = TimeLineText.GetPaddedFullRangeMinMax(df_m)


# Display one line:

is_draw_small = False                                           # Maintain day resolution to not lose small intervals.
is_draw_point_only = False
isVerbose = False
isClipEdge = True
growConst = 2

line_size = t_end - t_start

timelineText = TimeLineText(t_start, t_end, line_size, is_draw_small, is_draw_point_only, isVerbose, isClipEdge)

timelineText.DescribeScale()

idx = 194

l = timelineText.GetTargetInterval(df_m, idx)

target_begin = l[0]
target_end   = l[1]- 1    # -1: Now including last given day

target_length = 1 + target_end - target_begin

print(f"idx = {idx}: [{target_begin}, {target_end}], L = {target_length}")

#
#
# Preliminary preprocessing:
#
#
# For one individual:
#
# Create intervals as above/ p3.py
#
# Rasterize, see overlaps. Take note of warnings.
# For now - discard overlapping.
#
#
# Find air distances.
#
# Take note of small distances. (1,2,3,4,5)
#
# For now discard air gap cases.

































