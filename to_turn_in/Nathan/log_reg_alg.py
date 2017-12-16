import math
import numpy as np

def log_reg (x, y, w, rate=0.1, tradeoff=0.1):
    t = 0.0
    with np.errstate(over='raise', divide='raise'):
        for i in range(x.shape[0]):
            if y[i] * np.dot(x[i], w) <= 0:
                try:
                    w = w + (rate / (1.0 + t)) * np.reshape(((x[i] * y[i]) / (1 + math.e**(y[i] * np.dot(x[i], w)))), (x[i].shape[0], 1)) - (2 * w) / tradeoff ** 2
                except:
                    import pdb; pdb.set_trace()
            t += 1
    return w
