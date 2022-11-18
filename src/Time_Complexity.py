"""
Used this module to generate data
"""

# import essential modules
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# X = 2.235 * (np.random.random_integers(999, size=(10000, 3000)) - 1) / 4.


# generate an empty dataframe that we will use later to train our regressor
df = pd.DataFrame(columns=['m', 'n', 'kernel', 'n_iter','training_time'])


# create a loop to collect data
for i in range(10, 1001, 10):                                 # loop over the number of training instances (m)
    for k in range(3, 31, 3):                                   # loop over the number of features  (n)
        for l in range(0, 2):                                   # loop over 2 types of kernels linear & rbf 
            x = np.random.normal(size=(i, k))                   # generate x as float from normal distribution with size (m, n)
            y = np.random.randint(2, size=(i,))                 # generate y as int between 0 & 1 size(m,)
            if l == 0:                                          # if kernel is 'linear'               
                our_time = []                                   # empty list that will hold training time of 5 runs (we will take average)
                n_iter_l = []                                   # samewise for n_iter
                for a in range(5):                                  
                    clf = SVC(kernel='linear', random_state=8)      # insantiate classifier with linear kernel 
                    start_time = time.process_time()                # start time recording
                    clf.fit(x, y)                                   # train
                    end_time = time.process_time()                  # end time recording
                    our_time_a = end_time - start_time              # calculate time 
                    our_time.append(our_time_a)                     # append the time to the above list
                    n_iter_l.append(clf.n_iter_[0])                 # append the n_inter to the above list
            else:
                our_time = []
                n_iter_l = []
                for a in range(5):
                    clf = SVC(kernel='rbf', random_state=8)         # insantiate classifier with rbf kernel
                    start_time = time.process_time()                # start time recording
                    clf.fit(x, y)
                    end_time = time.process_time()                  # end time recording
                    our_time_a = end_time - start_time              # calculate time 
                    our_time.append(our_time_a)
                    n_iter_l.append(clf.n_iter_[0])
            
            our_time_average = sum(our_time)/len(our_time)          # get the average training time
            n_iter = sum(n_iter_l)//len(n_iter_l)                   # get the average number of iterations

            # append the info that we need to the df
            df = pd.concat([pd.DataFrame([[x.shape[0], x.shape[1], l, n_iter, our_time_average]], columns=df.columns), df])

        print(f'you reached m of size {i}, n of size {k}')
    print(f'you Finished m of size {i}')

# reset the index
df.reset_index(drop=True, inplace=True)

# transform to csv
df.to_csv('/home/lina/PycharmProjects/Project1/Data/data10_1000.csv')


