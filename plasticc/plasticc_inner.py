

####################################################################################
#
#    process_single_item_inner0()
#

def process_single_item_inner0(df, idx_begin, idx_end):
    
    # an_id = df.object_id.values[idx_begin:idx_end]

    ar_mjd = df.mjd.values[idx_begin:idx_end]
    au_passband = df.passband.values[idx_begin:idx_end]

    ar_flux = df.flux.values[idx_begin:idx_end]
    ar_flux_err = df.flux_err.values[idx_begin:idx_end]

    ais_detected = df.detected.values[idx_begin:idx_end]

    # assert an_id.min() == an_id.max()

    #assert 4 == 5

    return an_id.shape[0]
"""c"""

# Early stage load/preprocess: Sort by passband after object, then mjd.

# Append counter for new passband? Ref new object et.c.

au_passband =  np.array([   0,     0,     1,     1,     2,    3], dtype = np.uint8)

ar_mjd =       np.array([900.0, 901.2, 930.0, 931.9, 941.0, 950.3], dtype = np.float32)
ais_detected = np.array([True,   True, False,  False, False, True], dtype = np.bool)
ar_flux =      np.array([ 3.4,    2.1,   0.5,  -0.4,  2.0,    2.1], dtype = np.float32)
ar_flux_err =  np.array([ 0.4,    0.2,   0.1,   0.1,  0.3,    0.4], dtype = np.float32)


ar_mjd -= np.min(ar_mjd)

all_bands = np.array([0, 1, 2, 3])

start_indinces = np.searchsorted(au_passband, all_bands, side = 'left')

stop_indices = start_indinces[1:]
stop_indices = np.append(stop_indices, au_passband.shape[0])

for b, e in zip (start_indinces, stop_indices):
    print (b, e)


def single_passband(ar_mjd, ais_detected, ar_flux, ar_flux_err):

    slot_x = [0.0, 1.0, 2.0, 3.0]
    y_out = np.interp(slot_x, ar_mjd, ar_flux, left=None, right=None, period=None)







