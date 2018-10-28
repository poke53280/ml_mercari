

####################################################################################
#
#    process_single_item()
#

def process_single_item_inner0(df, idx_begin, idx_end):
    
    an_id = df.object_id.values[idx_begin:idx_end]

    ar_mjd = df.mjd.values[idx_begin:idx_end]
    au_passband = df.passband.values[idx_begin:idx_end]

    ar_flux = df.flux.values[idx_begin:idx_end]
    ar_flux_err = df.flux_err.values[idx_begin:idx_end]

    ais_detected = df.detected.values[idx_begin:idx_end]

    assert an_id.min() == an_id.max()

    #assert 4 == 5

    return an_id.shape[0]
"""c"""




