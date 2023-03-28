import numpy as np
from bootstraping_confidence_intervals import bootstraping_confidence_intervals

y_true = np.random.choice(2, 100)
y_pred = np.random.choice(2, 100)

print(bootstraping_confidence_intervals(y_true, y_pred))