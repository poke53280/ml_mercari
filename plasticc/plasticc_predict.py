


# Assume AUC 0.87

from sklearn.metrics import log_loss

import numpy as np

y_true = np.array([[0, 1, 0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0]])

y_pred = np.array([[0.0, 0.8, 0.1, 0.1,0,0,0,0,0,0], [0.1, 0.1, 0.8, 0.0,0,0,0,0,0,0] ])

log_loss(y_true, y_pred)

