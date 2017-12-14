import pdb

from math import log

class Bayes:
    def __init__(self):
        self.total = 0.0
        self.ones = 0.0
        self.neg_ones = 0.0
        self.one_given_one = []
        self.one_given_neg_one = []
        self.zero_given_one = []
        self.zero_given_neg_one = []

    def predict(self, x, j):
        pdb.set_trace()
        prob_one, prob_neg_one = log(self.ones / self.total), log(self.neg_ones / self.total)
        for i in range(x[j].size):
            
            if x[j,i] == 1:
                prob_one += log(self.one_given_one[i])
                prob_neg_one += log(self.one_given_neg_one[i])
            elif x[j,i] == 0:
                prob_one += log(self.zero_given_one[i])
                prob_neg_one += log(self.zero_given_neg_one[i])
            else:
                print('error3')
        if prob_one >= prob_neg_one:
            return 1
        return -1

def bayes(x, y, smoothing=1.0):
    ones, neg_ones = 0, 0
    for el in y:
        if el == 1:
            ones += 1
        elif el == -1:
            neg_ones += 1
        else:
            print('error0')
    b = Bayes()
    b.ones = float(ones)
    b.neg_ones = float(neg_ones)
    total = float(ones + neg_ones)
    b.total = total
    for j in range(x[0].size):
        if j % 1000 == 0:
            print('bayes training', j, 'of', x[0].size)
        one_given_one, one_given_neg_one = 0, 0
        zero_given_one, zero_given_neg_one = 0, 0
        for i in range(len(x)):
            if x[i,j] == 1:
                if y[i] == 1:
                    one_given_one += 1
                elif y[i] == -1:
                    one_given_neg_one += 1
                else:
                    print('error1')
            elif x[i,j] == 0:
                if y[i] == 1:
                    zero_given_one += 1
                elif y[i] == -1:
                    zero_given_neg_one += 1
                else:
                    print('error2')
        prob_one_given_one = float(one_given_one + smoothing) / float(b.ones + 2.0 * smoothing)
        prob_one_given_neg_one = float(one_given_neg_one + smoothing) / float(b.neg_ones + 2.0 * smoothing)
        prob_zero_given_one = float(zero_given_one + smoothing) / float(b.ones + 2 * smoothing)
        prob_zero_given_neg_one = float(zero_given_neg_one + smoothing) / float(b.neg_ones + 2.0 * smoothing)
        b.one_given_one.append(prob_one_given_one)
        b.one_given_neg_one.append(prob_one_given_neg_one)
        b.zero_given_one.append(prob_zero_given_one)
        b.zero_given_neg_one.append(prob_zero_given_neg_one)
    return b
    
