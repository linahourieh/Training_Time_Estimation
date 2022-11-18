
from sklearn.svm import SVC
import time
import numpy as np
np.random.seed(0)


### get a sample dataof specific dimension
svc_data_x = np.random.randn(3090, 9)
svc_data_y = np.random.randint(3090, size=(3090,))
###
clf = SVC(kernel='rbf', random_state=7)

### train and record the time
def train_algo():
    start_time = time.process_time()
    clf.fit(svc_data_x, svc_data_y)
    end_time = time.process_time()
    return start_time, end_time

# print out the time & number of iteration
start_time, end_time = train_algo()
our_time = end_time - start_time
print(our_time)
print('iter',sum(clf.n_iter_)/len(clf.n_iter_))



#
#import mlflow
#logged_model = 'runs:/6087f5d015d44a6f8ea05353f7e9d0bc/regressor'

# Load model as a PyFuncModel.
##loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
#import pandas as pd

#df = pd.DataFrame(columns=['m', 'n', 'kernel', 'n_iter'])

#df = pd.concat([pd.DataFrame([[1000, 20, 1, 3000]], columns=df.columns), df])

#print(loaded_model.predict(df))
#



