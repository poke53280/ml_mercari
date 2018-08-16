

import sys
import os

import numpy as np

sys.path

cwd = os.getcwd()

os.chdir('C:\\Users\\T149900\\ml_mercari')

from general.TimeSlotAllocator import get_slot_data




class DataProviderImpl:

    # Time points, sorted from high to low for which data is available.
    def getTimeStamps(self):
        return [3,2,1, -4]

    # Number of float32 elements for each time point.
    def getDataPerElement(self):
        return 2

    # Provide data for time point time_value adjusted with time offset valueOffset.
    def getTimeRecord(self, time_slot_value, alloc_location_list, data_size):


        data_points = self.getTimeStamps()

        for x in alloc_location_list:
            assert x >= 0 and x < len(self.getTimeStamps()), "time stamp index out of range"

            print(f" Requesting element idx = {x}, value = {data_points[x]}")
        

        an = np.zeros(data_size, dtype = np.float32)

        return an

"""c"""

    
b = DataProviderImpl()

b.getTimeRecord(3, [0,3], 2)


values = [90, 80, 4, 3, -1]
slots = [40, 40, 3, 3]


slot_allocator_best_value(values, slots, 5, True)

slot_allocator_multi_value(values, slots, 5)


b.getTimeStamps()


d = get_slot_data(b, [40, 40, 3, 3], 5, True)



# Find nearest value in slot for values
# First make work on ascending order;

# Slots must be unique valued

import bisect

values = [-1, 3, 4, 4.9,5.5, 6, 6, 7]
slots = [3,4,5, 6]



def slot_allocator_multi_value(values, slots, rTolerance):

    s_res = []

    for s in slots:
        s_res.append([])

    bisect_lo = 0


    for idx, value in enumerate(values):

        bisect_lo = bisect.bisect_left(slots, value, lo = bisect_lo, hi = len (slots))

        if bisect_lo == len(slots):
            slot_idx = len(slots) -1
        else:
            slot_idx = bisect_lo

        slot_value = slots[slot_idx]


        if np.abs(value - slot_value) > rTolerance:
            pass
        else:
            s_res[slot_idx].append(idx)


    return s_res    

"""c"""
    

slot_allocator_multi_value(values, slots, 0.01)

     






    slots = [-1,-2, -3, -4, -5.1, -6, -7, -8, -9]
    rTolerance = 0.9

    data = get_slot_data(b, slots, rTolerance, False)

    print (data)

"""c"""

# DataProviderImpl provides data at times 3, 2, 1, -4

#


TimeSlotAllocator_example()

