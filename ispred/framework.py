
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 500)
# pd.set_option('display.min_rows', 100)


np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=200)
np.core.arrayprint._line_width = 480

pd.options.display.float_format = '{:.1f}'.format



####################################################################################
#
#   create_input
#

def create_input():

    l_fnr_id = [11111111111,11111111111, 22222222222, 22222222222, 22222222222, 0, 33333333333, 4444444444]
    l_act_id = [3000000000000, 3000000000000, 4000000000000, 4000000000000, 4000000000000, 4000000000000, 6000000000000, 7000000000000]

    l_fom = [18100, 18101, 18120, 18140, 18150, 18150, 18155, 18170 ]
    l_tom = [18140, 18160, 18120, 18190, 18155, 18188, 18156, 18210 ]

    l_create_time = [18100.1, 18101.1, 18120.9, 18140.2, 18150.3, 18150.4, 18155.0, 18170.7 ]


    df_in = pd.DataFrame({'fnr' : l_fnr_id, 'aktorid': l_act_id, 'request_created': l_create_time, 'tilfelle_start_date': l_fom, 'tilfelle_end_date': l_tom})

    return df_in



####################################################################################
#
#   create_output
#

def create_output(df_in):

    df_output = df_in.copy()

    df_output['prediksjon_created'] = [18120.9, 18121.2, 18121.2, 18123.9, 18123.9, 18123.9, 18123.9, 18123.9]

    df_output['c_datastate'] = ['ok', 'ok', 'ok', 'none', 'ok',  'too_little', 'large_discrepancy',  'ok']

    df_output['div'] = "json/xml-txt"

    df_output['dr_prediksjon'] = [12.0, 9.0,  3.0,  np.nan, 21.0, np.nan,  np.nan,  19.0]

    return df_output





