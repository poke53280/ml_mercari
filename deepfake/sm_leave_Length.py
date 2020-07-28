
import numpy as np
import pandas as pd


l_dnreg = [18304, 18338, 18346]
l_dfom = [18309, 18338, 18358]
l_dtom = [18322, 18357, 18382]

df = pd.DataFrame({'d_reg': l_dnreg, 'd_fom' : l_dfom, 'd_tom': l_dtom})

# Is a leave complete, or will there be extensions?


# Information available
d_reg  d_fom  d_tom
18304  18309  18322

# We are at time = 18304. The sm is just received. Is the leave complete?
# Complete = There will be no further messages joining the last leave, max gap three days.

###############################################################################################

Time

18302
18303 0
18304 pr  SM:  18309 => 18322
18305 pr
18306 pr
18307 pr
18308 pr
18309 sm
18310 sm
18311 sm
18312 sm
...
18330 sm
18331 sm
18332 sm
18333 0
18334 0
18335 0




... END OF MESSAGES (WAIT ~ 50)


#########################################################


Time 
18338 sm SM:  18338 => 18357
18339 sm
18340 sm
18341 sm
...
18343 sm
18344 sm
18345 sm
18346 sm SM:  18358  => 18382. New message relative to last end: 18346 - 18357 = -11 
18347 sm
18348 sm
13449 sm
...
18382 sm
18383 0
18384 0
18385 0

     ...END OF MESSAGES (WAIT ~ 50)


######################################################


Time
18316:   SM:  18316 => 18334
18317
18318
18319
18320
18321
18322
18323
18324
18325
18326
18327
...
18331
18332
18333:   SM:  18335 => 18352
18334
18335
18336
..
18351
18352  Last day of sm
18353
18354
18355
...
...END OF MESSAGES (WAIT ~ 50)


