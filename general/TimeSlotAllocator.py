
import numpy as np

# Values are input as sorted from big to small.
# Slots are input with target values from big to small.
#
# Returned list sized like input slots. Values are indices into the sorted value array. -1 if no value found to match.
# Tolerance is max appected difference between condidate data value and slot target value for insertion to slot.
# Slots are filled greedily - high values first.
#
# See example() for usage


def slot_allocator(data, slots, tolerance, isVerbose):
    
    assert data == sorted(data, reverse=True)
    assert slots == sorted(slots, reverse=True)

    filled_slot = []

    current_slot_idx = 0
    current_value_idx = 0

    while len(filled_slot) < len (slots):

        isOutOfValues = current_value_idx >= len (data)
        isOutOfSlots = current_slot_idx >= len(slots)

        if isOutOfValues or isOutOfSlots:
            filled_slot.append(-1)
            if isVerbose: print("value or slot out of bonds, pad with -1")
            continue

        current_value = data[current_value_idx]
        current_slot = slots[current_slot_idx]

        if isVerbose: print(f"val = {current_value}, idx = {current_slot}")

        isValueTooLarge = (current_value - current_slot) > tolerance
        isValueTooSmall = (current_slot - current_value) > tolerance

        if isValueTooLarge:
            if isVerbose: print("Value too large, going to next")
            current_value_idx = current_value_idx + 1

        elif isValueTooSmall:
            if isVerbose: print("Slot too large, setting empty going to next")
            filled_slot.append(-1)
            current_slot_idx = current_slot_idx + 1

        else:
            if isVerbose: print("Match")
            filled_slot.append(current_value_idx)
            current_slot_idx = current_slot_idx + 1
            current_value_idx = current_value_idx +1 

    return filled_slot

"""c"""

##############################################################################
#
#    get_slot_data()
#
#

def get_slot_data(data_provider, list_time_slots_configuration, rTimeTolerance, isVerbose):

    assert hasattr(data_provider, 'getTimeStamps'), "data_provider must support getTimeStamps()"
    assert hasattr(data_provider, 'getTimeRecord'), "data_provider must support getTimeRecord()"
    assert hasattr(data_provider, 'getDataPerElement'), "data_provider must support getDataPerElement()"

    data_per_element = data_provider.getDataPerElement()

    n_slots = len (list_time_slots_configuration)

    data = np.empty(n_slots * data_per_element, dtype = np.float32)

    data.fill(np.nan)

    list_time_values = data_provider.getTimeStamps()

    l_slot_allocation = slot_allocator(list_time_values, list_time_slots_configuration, rTimeTolerance, isVerbose)

    assert len(l_slot_allocation) == n_slots, "slot allocater bad return"

    for i, alloc_location in enumerate(l_slot_allocation):
        iSlotIndex = i
        iDataIndex = alloc_location

        if iDataIndex == -1:
            if isVerbose: print(f"{iSlotIndex}: Empty")

        else:
            assert (iDataIndex >= 0) and (iDataIndex <= len(list_time_values)), "index out of range"

            timeSlotTargetValue = list_time_slots_configuration[iSlotIndex]
            timeActualValue = list_time_values[iDataIndex]
            valueOffset = timeActualValue - timeSlotTargetValue
            if isVerbose: print(f"{iSlotIndex}: Insert data idx = {iDataIndex}. Values {timeActualValue} => {timeSlotTargetValue}. value offset= {valueOffset}")

            # Request data provider to prepare item at iDataIndex to offset valueOffset
            data_record = data_provider.getTimeRecord(timeActualValue, valueOffset, data_per_element)

            assert data_record.shape[0] == data_per_element, "data record size incorrect"

            data_offset = iSlotIndex * data_per_element

            data[data_offset:data_offset + data_per_element] = data_record

    return data                

"""c"""



def example():
    values = [90, 80, 4, -1]
    slots = [40, 40, 3, 3]

    slot_allocator(values, slots, 5, False)
    # => [-1, -1, 2, 3]

