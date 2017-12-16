import pdb
import numpy as np

def perceptron(x, y, w, rate=0.1, margin=0, average=None, dynamic_learning_num=None, initial_learning_rate=None):
    training_updates = 0
    for i in range(x.shape[0]):
        if dynamic_learning_num is not None and initial_learning_rate is not None:
            learning_rate = initial_learning_rate / (1 + dynamic_learning_num)
            dynamic_learning_num += 1
        if average is not None:
            average = (average + w) / 2.0
        made_mistake = y[i] * np.dot(x[i], w) <= margin
        if made_mistake:
            update_vec = (rate * y[i] * x[i]).reshape(w.shape)
            w += update_vec
            training_updates += 1
    if average is None:
        return w, training_updates
    else:
        return average, training_updates
