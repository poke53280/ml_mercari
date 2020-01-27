
############################################
#
#   set_path_for_ide
# 

# Run in IDE:

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np


path_work = Path("C:\\Users\\T149900\\Documents\\GitHub")

path = path_work

assert path.is_dir()

path = path / "ml_mercari" / "deepfake"
assert path.is_dir()

sys.path.insert(0, str(path))

# pandas / numpy print options

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=200)
np.core.arrayprint._line_width = 480

