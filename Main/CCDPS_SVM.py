
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

def SVM_Linear():
	#Linear SVM Begins Here
	file_obj=open("Training_Dataset_short_headerless.csv", "r")
	raw_data=np.loadtxt(file_obj, delimiter=',')
	x_neg=np.empty(shape=[0,0])
	y_neg=np.empty(shape=[0,0])
	x_pos=np.empty(shape=[0,0])
	y_pos=np.empty(shape=[0,0])
	count=0
	c_p=0
	c_n=0
	for i in raw_data:
	    count+=1
	    # Checking if the label is 0
	    if i[-1]==0:
	        c_p+=1
	        if x_neg.shape[0]==0:
	            x_neg=np.array([i[5:-2]])
	        else:
	            x_neg=np.vstack((x_neg, [i[5:-2]]))
	        # Adding negative label as -1
	        if y_neg.shape[0] == 0:
	            y_neg=np.array([-1])
	        else:
	            y_neg=np.hstack((y_neg, -1))
        # Checking if the label is 1
	    elif i[-1]==1:
	        c_n+=1
	        if x_pos.shape[0]==0:
	            x_pos=np.array([i[5:-2]])
	        else:
	            x_pos=np.vstack((x_pos, [i[5:-2]]))
	        # Adding positive label as 1
	        if y_pos.shape[0]==0:
	            y_pos=np.array([1])
	        else:
	            y_pos=np.hstack((y_pos, 1))

	X=np.vstack((x_pos, x_neg))
	y=np.concatenate((y_pos,y_neg))

	# Initializing the parameters
	b=-10
	w=np.array([1,-1]).reshape(-1,1)
	m,n=X.shape
	y=y.reshape(-1,1)*1.
	X_quote=y*X
	H=np.dot(X_quote , X_quote.T)*1.

	# Initializing the solver variables
	P=cvxopt_matrix(H)
	q=cvxopt_matrix(-np.ones((m, 1)))
	G=cvxopt_matrix(-np.eye(m))
	h=cvxopt_matrix(np.zeros(m))
	A=cvxopt_matrix(y.reshape(1, -1))
	b=cvxopt_matrix(np.zeros(1))

	cvxopt_solvers.options['show_progress']=False
	cvxopt_solvers.options['abstol']=1e-10
	cvxopt_solvers.options['reltol']=1e-10
	cvxopt_solvers.options['feastol']=1e-10

	sol=cvxopt_solvers.qp(P,q,G,h,A,b)
	alphas=np.array(sol['x'])

	w=((y*alphas).T@X).reshape(-1,1)
	S=(alphas>1e-4).flatten()
	b=y[S]-np.dot(X[S], w)

	file_obj=open("Testing_Dataset_headerless.csv", "r")
	raw_data=np.loadtxt(file_obj, delimiter=',')
	x_test=[]
	y_test=0
	y_pred=[]
	for i in raw_data:
	    x_test=np.array([i[5:-2]])
	    if i[-1] > 0:
	        y_test=1
	    else:
	        y_test=0
	    
	    y_pred.append(np.sign(np.dot(x_test, w)+b[0])==y_test)
	return 100 * sum(y_pred)[0][0]/len(y_pred)
	
def SVM_Kernel():
	#Scikit-learn SVM Begins Here
	df_train=pd.read_csv("Training_Dataset_orig.csv")
	X_temp=df_train[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
	y_temp=df_train.label
	X_train_svm,X_test_svm,y_train_svm,y_test_svm=tts(X_temp,y_temp,test_size=0.3,random_state=42)

	df_test=pd.read_csv("Testing_Dataset_orig.csv")
	X_test_svm=df_test[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
	y_test_svm=df_test.label

	SVM_clf=SVC(kernel = 'rbf', gamma='auto')
	Grid_SVM_clf=GridSearchCV(estimator=SVM_clf, param_grid=dict(C=np.logspace(-6, -1, 10)), cv=5, n_jobs=-1)
	Grid_SVM_clf.fit(X_train_svm, y_train_svm)

	y_pred=Grid_SVM_clf.best_estimator_.predict(X_test_svm)
	#Scikit-learn SVM Ends Here
	return 100 * accuracy_score(y_test_svm,y_pred)

if __name__=='__main__':
	print(SVM_Linear())
	print(SVM_Kernel())