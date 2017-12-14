import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#import pdb; pdb.set_trace()
matches = pd.read_csv('past_matches_results.csv')
train, test = train_test_split(matches, test_size=0.1)
train.to_csv('past_matches_results_train.csv')
test.to_csv('past_matches_results_test.csv')

train = shuffle(train)
n = train.shape[0]
slice_size = n // 5
i = 0
for j in range(5):
    temp = train[i:i+slice_size]
    temp.to_csv('past_matches_results_train_cv_'+str(j)+'.csv')
    i += slice_size
    
