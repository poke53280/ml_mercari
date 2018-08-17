
import numpy as np
import bisect

# Values are input as sorted from small to big.
# Slots are input with target values from small to big.
#
# Returned list sized like input slots. Values are indices into the sorted value array. Sub lists may be empty - indicating empty slot.
# Tolerance is max appected difference between condidate data value and slot target value for insertion to slot.
# Slots are filled greedily - low values first.
#
# See example() for usage


# Warning: Never run code.
#

def slot_allocator_best_value(data, slots, tolerance, isVerbose):
    
    assert data == sorted(data, reverse=False), "data sorted small to big"
    assert slots == sorted(slots, reverse=False), "slots sorted small to big"

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

        if isValueTooSmall:
            if isVerbose: print("Value too small, going to next value")
            current_value_idx = current_value_idx + 1

        elif isValueTooLarge:
            if isVerbose: print("Slot too small, setting empty going to next")
            filled_slot.append(-1)
            current_slot_idx = current_slot_idx + 1

        else:
            if isVerbose: print("Match")
            filled_slot.append(current_value_idx)
            current_slot_idx = current_slot_idx + 1
            current_value_idx = current_value_idx +1 

    assert len(filled_slot) == len(slots)


    s_res = []

    for s in filled_slot:
        print(s)
        if s == -1:
            s_res.append([])
        else:
            s_res.append([s])

    return s_res

"""c"""



def slot_allocator_multi_value(values, slots, rTolerance):


    assert len (np.unique(slots)) ==  len (slots), "slots must be unique valued"

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

    # l_slot_allocation = slot_allocator_best_value(list_time_values, list_time_slots_configuration, rTimeTolerance, isVerbose)

    l_slot_allocation = slot_allocator_multi_value(list_time_values, list_time_slots_configuration, rTimeTolerance)

    assert len(l_slot_allocation) == n_slots, "slot allocater bad return"

    for i, alloc_location_list in enumerate(l_slot_allocation):
        iSlotIndex = i
        iDataIndexList = alloc_location_list

        if len (alloc_location_list) == 0:
            if isVerbose: print(f"{iSlotIndex}: Empty")

        else:

            for iDataIndex in alloc_location_list:
                assert (iDataIndex >= 0) and (iDataIndex <= len(list_time_values)), "index out of range"


            timeSlotTargetValue = list_time_slots_configuration[iSlotIndex]

            data_record = data_provider.getTimeRecord(timeSlotTargetValue, alloc_location_list)

            assert data_record.shape[0] == data_per_element, "data record size incorrect"

            data_offset = iSlotIndex * data_per_element

            data[data_offset:data_offset + data_per_element] = data_record

    return data                

"""c"""



def example():
    values = [90, 80, 4, -1]
    slots = [40, 40, 3, 3]

    slot_allocator_best_value(values, slots, 5, False)
    # => [-1, -1, 2, 3]



