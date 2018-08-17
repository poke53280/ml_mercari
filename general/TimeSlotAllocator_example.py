

import sys
import os

import numpy as np

sys.path

cwd = os.getcwd()

os.chdir('C:\\Users\\T149900\\ml_mercari')

from general.TimeSlotAllocator import *


class DataEntry:

    def __init__(self, a, b, c, d):
        self._a = a
        self._b = b
        self._c = c
        self._d = d


    def getKeyTime(self):
        return self._a



class DataProviderImpl:

    def addEntry(self, d):
        assert not d.getKeyTime() in self._d, f"Entry at key time = {d.getKeyTime()} already exists"
        self._d[d.getKeyTime()] = d


    def __init__(self):
        self._d = {}


    # Time points, sorted from high to low for which data is available.
    def getTimeStamps(self):
        return list (b._d.keys())


    # Number of float32 elements for each time point.
    def getDataPerElement(self):
        return 2


    def getData(self, l_requested_data_idx, offset):
        
        # 'count' and 'mean'
        assert len(l_requested_data_idx) > 0, "Requested data list is empty"

        l_requested_data = []

        for k in l_requested_data_idx:
            assert k in self._d, f"key {k} not in data dictionary"
            l_requested_data.append(self._d[k])


        sum_c = 0

        for v in l_requested_data:
            sum_c +=  (v._c - offset)
     

        anResult = np.empty(self.getDataPerElement(), dtype = np.float32 )

        anResult[0] = len (l_requested_data)
        anResult[1] = sum_c

        return anResult


    # Provide data for time point time_value adjusted with time offset valueOffset.
    def getTimeRecord(self, time_slot_value, alloc_location_list):

        assert len(alloc_location_list) > 0

        data_points = self.getTimeStamps()

        l_requested_data = []

        for x in alloc_location_list:
            assert x >= 0 and x < len(self.getTimeStamps()), "time stamp index out of range"
            l_requested_data.append(data_points[x])
            
        
        print(f" time slot at {time_slot_value}: Requesting elements {l_requested_data}")

        an = self.getData(l_requested_data, time_slot_value)

        assert an.shape[0] == self.getDataPerElement()

        return an

"""c"""


    
b = DataProviderImpl()

b.addEntry(DataEntry(-10,2,3,4))
b.addEntry(DataEntry(3,2,31,4))
b.addEntry(DataEntry(5,2,31,4))
b.addEntry(DataEntry(9,2,31,4))



slots = [-6,2,3,4]



d = get_slot_data(b, slots, 5, True)





