"""
Module for training differnet regressors to predict training time.
"""
# import essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
import pickle
import mlflow
import glob, os

# import data 
path = '/home/lina/PycharmProjects/Project1/Data' # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.concat((pd.read_csv(f) for f in all_files[::-1]), ignore_index=True)
df.drop(columns=['Unnamed: 0'], inplace= True)

# divide it into x & y
X = df.drop(columns=['training_time'])
y = df['training_time']


# this is only for polynomial regressor
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.25, random_state=42)

# split the data into train, test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# polynomial_features= PolynomialFeatures(degree=2)
# x_poly = polynomial_features.fit_transform(x)
# print(df)

reg = LinearRegression()
# reg = Ridge(random_state=9)
# reg = Lasso(random_state=9)
# reg = RandomForestRegressor(random_state=9)
# reg = LinearSVR(random_state=9, max_iter=1000000)
# reg = SGDRegressor(random_state=9)

def calculate_metrics(y_test,y_pred, y_train, y_pred_train):
    """
    Calculate metrics regardless of regressor
    """

    R2 = r2_score(y_test, y_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_train =  mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred)

    metric_dict = {'R_squared': R2,
                    'RMSE_train': rmse_train,
                    'RMSE_test': rmse_test,
                    'MAE_train': mae_train,
                    'MAE_test': mae_test}
    
    return metric_dict

with mlflow.start_run():  # start a mlflow run
    mlflow.log_params(reg.__dict__)  # log parameters
    reg.fit(X_train, y_train)  # train the algorithm
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)  # predict the outcome
    mlflow.sklearn.log_model(reg, 'Polynomial Regression')  # log the model into artifact
    metric_dict = calculate_metrics(y_test,y_pred, y_train, y_pred_train)
    mlflow.log_metrics(metric_dict)  # we only need metric dic for mlflow so log it as metric
    mlflow.log_param('Name','Polynomial Regression' )



# save the model to disk
#filename = '/home/lina/PycharmProjects/Carrots/model/Polynomial_model.pkl'
#joblib.dump(reg, filename)

#pickle.dump(reg, open(filename, 'wb'))
# loaded_model = joblib.load(filename)#
# result = loaded_model.score(X_test, y_test)
# print(result)
#print(df)

#new_data = pd.DataFrame(columns=['m', 'n', 'kernel'])
#new_data = pd.concat([pd.DataFrame([[2000, 4, 1]], columns=new_data.columns), new_data])

#new_data_poly = poly.fit_transform(new_data)
#new_data_pred = reg.predict(new_data_poly)
#print(new_data_pred)





