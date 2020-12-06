Compiling Instructions:

Install/Check the following libraries before executing:

1) Pandas
2) Numpy
3) CSV
4) Collections
5) Random
6) Math
7) Sklearn
8) Cvxopt

If the above libraries are available, directly run CCDPS_Demo.py from the Demo folder to run the algorithms on the toy datasets.

The main file can be run by executing CCDPS_Main.py from the Main folder to run the algorithms on the original datasets.

Common Points:
- All the accuracies listed in the report are based on the execution of CCDPS_Main.py.
- CCDPS_Main.py takes approximately 45 minutes to 1 hour to execute completely and show the output results.
- The correctness of the code can be verified by directly running CCDPS_Demo.py 
- The accuracies obtained by CCDPS_Demo.py might significantly differ from the original output results.

For KNN:
1.The dataset is already trimmed directly for the file to run
2.It takes roughly 30 min for the KNN to complete for the whole dataset
3.The same has been tested on SPYDER with python 3.7
4.k=7 gives best results . The program has been tested for k=1,3,5,7,9

For Logistic Regression and Gaussian Naive Bayes:
1.It runs directly on the original dataset and provides the output within seconds.

For SVM:
1.To execute the Linear SVM, please install the cvxopt library for python3 using either of the suitable commands:

apt install python3-cvxopt
conda install -c conda-forge cvxopt
pip install cvxopt

2.Linear SVM uses the Training_Dataset_short_headerless.csv as training data. The file is a minimized version of the original 
dataset upon which the model is trained. Testing_dataset_headerless.csv is also a minimized version of the original test set. 
For optimization the data has to be converted into the cvxopt_matrix which requires the numpy matrix and to avoid the issues 
related to type conversion, the headers were removed from the file.
3.The Scikit-learn's SVM with RBF kernel uses the Training_Dataset_orig.csv and Testing_Dataset_orig.csv. These are handled 
and processed using the pandas library as the Scikit functions support the pandas dataframes as inputs.
