

from benedict import benedict


from benedict import BeneDict

d = BeneDict()

d['profile', 'firstname'] = 'Fabio'
d['profile', 'lastname'] = 'Caccamo'

d
from benedict.dicts.parse import ParseDict


p = {'id': 90000, 'sm' : {'med': {'d0': 'a04'}}}




d[('sm','med')]




# set values by keys list
d['profile', 'firstname'] = 'Fabio'
d['profile', 'lastname'] = 'Caccamo'
print(d) # -> { 'profile':{ 'firstname':'Fabio', 'lastname':'Caccamo' } }
print(d['profile']) # -> { 'firstname':'Fabio', 'lastname':'Caccamo' }