from preprocessing import *
from implementations import *
from Final_CrossValidation import *

##Best Degree Settings
degree_0 = 2
degree_1 = 3
degree_2 = 3

##Best parameter setting for Logistic Regression 
gamma_0 = 1e-6
max_iters_0 = 15000

##Best parameter setting for Ridge Regression
lambda_1=[0]
lambda_2=[0]

##Data Splitting 
X0, X1, X2, indices0, indices1, indices2 = separate_data(X)

##Return y vector back to [-1,1] to be used with ridge regression
y1 = y[indices1]
y_s1=np.ones(len(y1))
y_s1[np.where(y1==0)]=-1
y2 = y[indices2]
y_s2=np.ones(len(y2))
y_s2[np.where(y2==0)]=-1

## Create Feature Transformation Matrix
phi_X0 = build_poly_cos_sin_poly(X0, degree_0, np.array(range(1,19)))
phi_X1 = build_poly_cos_sin_poly(X1, degree_1, np.array(range(1,23)))
phi_X2 = build_poly_cos_sin_poly(X2, degree_2, np.array(range(1,30)))


##Obtain Best weights
w0=reg_logistic_regression(y0, phi_X0, 0, gamma_0, max_iters_0)
w1=cross_validation_demo_ridge_regression(y_s1,phi_X1,4,[0],seed=250)
w2=cross_validation_demo_ridge_regression(y_s2,phi_X2,4,[0],seed=250)

##Load Testing Data Set
DATA_TEST_PATH = '../test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
X0_te, X1_te, X2_te, indices0_te, indices1_te, indices2_te = separate_data(tX_test)
OUTPUT_PATH = '../woohoo.csv'
ids_test0 = ids_test[indices0_te]
ids_test1 = ids_test[indices1_te]
ids_test2 = ids_test[indices2_te]


##Apply Feature Transformation on Test Set
phi_X0_te = build_poly_cos_sin_poly(X0_te, degree_0, np.array(range(1,19)))
phi_X1_te = build_poly_cos_sin_poly(X1_te, degree_1, np.array(range(1,23)))
phi_X2_te = build_poly_cos_sin_poly(X2_te, degree_2, np.array(range(1,30)))

##Obtain Predictions
pred0 = predict_labels_log_regression(w0, phi_X0_te)
pred1 = predict_labels(w1, phi_X1_te)
pred2 = predict_labels(w2, phi_X2_te)

##Regroup
predictions = np.append(pred0, np.append(pred1, pred2))
ids = np.append(ids_test0, np.append(ids_test1, ids_test2))

##Create Submission
create_csv_submission(ids, predictions, OUTPUT_PATH)