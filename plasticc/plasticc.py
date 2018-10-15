
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d

local_dir = os.getenv('LOCAL_PY_DIR')

assert local_dir is not None, "Set environment variable LOCAL_PY_DIR to parent folder of ai_lab_datapipe folder. Instructions in code."

# * 'Windows/Start'button
# * Type 'environment...'. 
# * Select option 'Edit environment variables for your account' (NOT: 'Edit the system environment variables)
# * New. LOCAL_PY_DIR. Value is parent folder to ai_lab_datapipe
# * Restart IDE/python context. 

print(f"Local python top directoy is set to {local_dir}")
os.chdir(local_dir)



from ai_lab_datapipe.sf_pipeline.TimeLineTool import *


DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)



training = pd.read_csv(DATA_DIR + 'training_set.csv')

meta_training = pd.read_csv(DATA_DIR + 'training_set_metadata.csv')



training.sample(5)

meta_training.sample(5)


#ddf
m = meta_training.ddf == 1

q = meta_training[m]

len (np.unique(meta_training.object_id))

m = training.object_id == 207413

q = training[m]

q = q.sort_values(by = ['passband', 'mjd'])

m = q.passband == 0

q = q[m]

nCut = 5

x_mjd = np.array(q.mjd)


m = np.diff (x_mjd) < 2

m

np.where(m)



l_chunks = []

nOffset = 0



acData = np.array([1,2,31,33])

a = TimeLineTool_GetOptimalGroupSize(acData, False, 2)



#LOOK AT KDESTUDY kernel density.




TimeLineTool_Analyze_Cluster(acData, 3)


t = TimeLineText()





y_flux = np.array(q.flux)

y_std = np.array(q.flux_err)

x_mjd = x_mjd[:5]
y_flux = y_flux[:5]
y_std = y_std[:5]

def get_flux_sample(x_mjd, y_flux, y_std):
    assert x_mjd.shape[0] == y_flux.shape[0] == y_std.shape[0], "inequal array lengths"

    y_out = np.empty(shape=y_flux.shape[0], dtype = float)

    for idx in range (x_mjd.shape[0]):

        flux = np.random.normal(y_flux[idx], y_std[idx])
        y_out[idx] = flux

    return y_out


y_out = get_flux_sample(x_mjd, y_flux, y_std)

f = interp1d(x_mjd, y_out)
f2 = interp1d(x_mjd, y_out, kind='cubic')


xnew = np.linspace(x_mjd.min(), x_mjd.max(), num=41, endpoint=True)


import matplotlib.pyplot as plt

plt.plot(x_mjd, y_out, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')

plt.legend(['data', 'linear', 'cubic'], loc='best')

plt.show()


f2(x_mjd[0] + 0.01)