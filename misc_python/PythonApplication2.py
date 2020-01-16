

import dowhy

import pandas as pd

from dowhy.do_why import CausalModel
from IPython.display import Image, display

import dowhy.datasets






z= [i for i in range(10)]

random.shuffle(z)

df = pd.DataFrame(data = {'Z': z, 'X': range(0,10), 'Y': range(0,100,10)})


df


dir = "C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\"

# With GML file
model = CausalModel(data = df, treatment='X', outcome='Y', graph= dir + "test.gml")




model



