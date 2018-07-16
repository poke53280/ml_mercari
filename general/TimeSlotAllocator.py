
# Values are input as sorted from big to small. Fitting large values has the priority

# Slots are given with target values from big to small.

# Returned list sized like input slots, with indices into the sorted value array. -1 if no value found to match.
# tolerance is max appected difference between condidate data value and slot target value for insertion to slot.
# slots are filled greedily igh values first.
#

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


def example():
    values = [90, 80, 4, -1]
    slots = [40, 40, 3, 3]

    slot_allocator(values, slots, 5, True)
    # => [-1, -1, 2, 3]

