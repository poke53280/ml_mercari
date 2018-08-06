

import sys
import os

import numpy as np

sys.path

cwd = os.getcwd()

os.chdir('C:\\Users\\T149900\\ml_mercari')

from general.TimeSlotAllocator import get_slot_data


def TimeSlotAllocator_example():

    class DataProviderImpl:

        # Time points, sorted from high to low for which data is available.
        def getTimeStamps(self):
            return [3,2,1, -4]

        # Number of float32 elements for each time point.
        def getDataPerElement(self):
            return 2

        # Provide data for time point time_value adjusted with time offset valueOffset.
        def getTimeRecord(self, time_value, valueOffset, data_size):
            assert time_value in [3,2,1,-4]

            assert data_size == self.getDataPerElement(), "data_size == CreditCardBalanceRecord.staticGetDataSize()"

            # rec = ...

            an = np.zeros(data_size, dtype = np.float32)

            return an

    """c"""

    b = DataProviderImpl()

    b.getTimeRecord(-4, 0, 2)

    slots = [-1,-2, -3, -4, -5.1, -6, -7, -8, -9]
    rTolerance = 0.9

    data = get_slot_data(b, slots, rTolerance, False)

    print (data)

"""c"""




