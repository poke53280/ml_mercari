
import pandas as pd


df = pd.DataFrame.from_dict(
   {
       'col1': [1, 2],
       'col2': [pd.DataFrame.from_dict(
           {'inner_col1': ['one', 'two', 'three'],
           'inner_col2': ['four', 'five', 'six']}),
                pd.DataFrame.from_dict(
           {'inner_col1': ['seven', 'eight', 'nine'],
           'inner_col2': ['ten', 'eleven', 'twelve']})
               ]
   }
)

s
