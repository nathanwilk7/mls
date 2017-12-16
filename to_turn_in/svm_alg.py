import numpy as np

def svm(x, y, w, rate=0.1, regularization=0.1):
    t = 0.0
    for i in range(x.shape[0]):
        if y[i] * np.dot(x[i], w) <= 1:
            w = (1.0 - (rate / (1.0 + t))) * w + regularization * rate * np.reshape(y[i] * x[i], (x[i].shape[0], 1))
        else:
            w = (1.0 - (rate / (1.0 + t))) * w
        t += 1
    return w
