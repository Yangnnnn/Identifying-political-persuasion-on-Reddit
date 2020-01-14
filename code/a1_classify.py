from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import csv
from sklearn.model_selection import KFold
from scipy import stats

import warnings
#to ignore convergence warning
warnings.filterwarnings('ignore')

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    total = np.sum(C)
    correct=0
    for i in range(len(C)):
        correct = correct+C[i][i]
    if total != 0:
        acc = correct/total
    else:
        acc = 0
    return acc
    
    
        
def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    lst1 = []
    lst2 = []
    result = []
    for i in range(len(C)):
        lst1.append(C[i][i])
        lst2.append(np.sum(C[i]))
    
    for i in range(len(C)):
        if lst2[i] !=0:
            result.append(lst1[i]/lst2[i])
        else:
            result.append(0)
    return result
    
        
def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    lst1 = []
    lst2 = []
    result = []    
    for i in range(len(C)):
        lst1.append(C[i][i])
        lst2.append(np.sum(C[:,i])) 
    for i in range(len(C)):
        if lst2[i] != 0 :
            result.append(lst1[i]/lst2[i])
        else:
            result.append(0.0)
    return result
    

    

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print("Doing class31")
    iBest = 0
    max_acc = 0
    line1 = [1]
    line2 = [2]
    line3 = [3]
    line4 = [4]
    line5 = [5]
    #load file and array
    data = np.load(filename)
    data = data['arr_0']
    #get x and y then split them by using train_test_split 
    X, y = data[:,0:-1], data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #1.SVC: support vector machine with a linear kernel.
    clf_linear = SVC(kernel='linear',max_iter=1000)
    clf_linear.fit(X_train, y_train)
    #Perform classification on samples in X.
    y_pred_linear = clf_linear.predict(X_test)
    svc_linear_matrix = confusion_matrix(y_test, y_pred_linear)
    iBest = 1
    max_acc = accuracy(svc_linear_matrix)

    #2.SVC: support vector machine with a radial basis function (γ = 2) kernel.
    clf_rbf = SVC(kernel='rbf',gamma=2,max_iter=1000)
    clf_rbf.fit(X_train, y_train)
    y_pred_rbf = clf_rbf.predict(X_test)
    svc_rbf_matrix = confusion_matrix(y_test, y_pred_rbf)
    if accuracy(svc_rbf_matrix) > max_acc:
        iBest = 2
        max_acc = accuracy(svc_rbf_matrix)
    #3.RandomForestClassifier: with a maximum depth of 5, and 10 estimators.
    clf_forest = RandomForestClassifier(n_estimators=10, max_depth=5)
    clf_forest.fit(X_train,y_train)
    y_pred_forest = clf_forest.predict(X_test)
    forest_matrix = confusion_matrix(y_test, y_pred_forest)
    if accuracy(forest_matrix) > max_acc:
        iBest = 3  
        max_acc = accuracy(forest_matrix)
    #4.MLPClassifier: A feed-forward neural network, with α = 0.05.
    clf_mlp = MLPClassifier(alpha=0.05)
    clf_mlp.fit(X_train,y_train)
    y_pred_mlp = clf_mlp.predict(X_test)
    mlp_matrix = confusion_matrix(y_test,y_pred_mlp)
    if accuracy(mlp_matrix) > max_acc:
        iBest = 4  
        max_acc = accuracy(mlp_matrix)
    
    #5.AdaBoostClassifier: with the default hyper-parameters.
    clf_ada = AdaBoostClassifier()
    clf_ada.fit(X_train,y_train)
    y_pred_ada = clf_ada.predict(X_test)
    ada_matrix = confusion_matrix(y_test,y_pred_ada)
    if accuracy(ada_matrix) > max_acc:
        iBest = 5 
        max_acc = accuracy(ada_matrix)
    
    #save result to a csv file
    
    line1.append(accuracy(svc_linear_matrix))
    line2.append(accuracy(svc_rbf_matrix))
    line3.append(accuracy(forest_matrix))
    line4.append(accuracy(mlp_matrix))
    line5.append(accuracy(ada_matrix))
    

    line1 = line1 + recall(svc_linear_matrix)
    line2 = line2 + recall(svc_rbf_matrix)
    line3 = line3 + recall(forest_matrix)
    line4 = line4 + recall(mlp_matrix)
    line5 = line5 + recall(ada_matrix)
        
    
    line1 = line1 + precision(svc_linear_matrix)
    line2 = line2 + precision(svc_rbf_matrix)
    line3 = line3 + precision(forest_matrix)
    line4 = line4 + precision(mlp_matrix)
    line5 = line5 + precision(ada_matrix) 
                               
    for i in range(len(svc_linear_matrix)):
        line1 = line1 + list(svc_linear_matrix[i])
        line2 = line2 + list(svc_rbf_matrix[i])
        line3 = line3 + list(forest_matrix[i])
        line4 = line4 + list(mlp_matrix[i])
        line5 = line5 + list(ada_matrix[i])
        
    with open( 'a1_3.1.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(line1)
        writer.writerow(line2)
        writer.writerow(line3)
        writer.writerow(line4)
        writer.writerow(line5)
    print(iBest)
    
    print("Class31 done!")

    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    accuracies=[]
    print("Doing class32")
    if iBest == 1:
        clf = SVC(kernel='linear',max_iter=1000)
    if iBest == 2:
        clf = SVC(kernel='rbf',gamma=2,max_iter=1000)
    if iBest == 3:
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    if iBest == 4:
        clf = MLPClassifier(alpha=0.05)
    if iBest == 5:
        clf = AdaBoostClassifier()
    for i in [1000,5000,10000,15000,20000]:
        if i == 1000:
            X_1k = X_train[:i]
            y_1k = y_train[:i]               
        new_X_test = X_test[:i] 
        new_y_test = y_test[:i]
        new_X_train = X_train[:i]
        new_y_train = y_train[:i]
        clf.fit(new_X_train, new_y_train)
        y_pred = clf.predict(new_X_test)
        matrix = confusion_matrix(new_y_test,y_pred)
        accuracies.append(accuracy(matrix))
        
        
    with open( 'a1_3.2.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(accuracies)    
        
    print("Class32 done!")
    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print("Doing class33")
    k=[5,10,20,30,40,50]
    line1 =[]
    line2 =[]
    line3 =[]
    line4 =[]
    line5 =[]
    line6 =[]
    line7 =[]
    line8 =[]
    #3.3.1
    # I tried to use cols_32k = selector.get_support(indices=True) 
    # Then get selector.pvalues_[cols_32k]
    # I found that result is same as what I do (sort pvalues then convert it to a list and find first K p values)
    for j in k:
        selector = SelectKBest(f_classif, j)
        new_X = selector.fit_transform(X_train, y_train)
        pp = np.sort(selector.pvalues_)
        pp = pp.tolist()
        
        if j == 5:
            line1.append(5)
            line1 = line1 + pp[:j]
            
            # 3.3.2
            if i == 1:
                clf = SVC(kernel='linear',max_iter=1000)
            if i == 2:
                clf = SVC(kernel='rbf',gamma=2,max_iter=1000)
            if i == 3:
                clf = RandomForestClassifier(n_estimators=10, max_depth=5)
            if i == 4:
                clf = MLPClassifier(alpha=0.05)
            if i == 5:
                clf = AdaBoostClassifier()
                
            
            # for 1K part
            new_X = selector.fit_transform(X_1k, y_1k)
            clf.fit(new_X,y_1k)
            y_pred_1 = clf.predict(selector.transform(X_test))
            matrix_1 = confusion_matrix(y_test,y_pred_1)
            line7.append(accuracy(matrix_1))
            
            
           # 3.3.3 (a) get index of 5 features 
            cols_1k = selector.get_support(indices=True)
            print(cols_1k)
            
            
            
            # for 32K part
            new_X = selector.fit_transform(X_train, y_train)
            clf.fit(new_X,y_train)   
            y_pred_32 = clf.predict(selector.transform(X_test))
            matrix_32 = confusion_matrix(y_test,y_pred_32)
            line7.append(accuracy(matrix_32))
            
            #3.3.3 (a) get index of 5 features 
            cols_32k = selector.get_support(indices=True)
            print(cols_32k)

            #3.3.3(a) find common features
            
            line8.append(list(set(cols_1k) & set(cols_32k)))
            
            
            
        if j == 10:
            line2.append(10)
            line2 = line2 + pp[:j]
            
        if j == 20:
            line3.append(20)
            line3 = line3 + pp[:j]
            
        if j == 30:
            line4.append(30)
            line4 = line4 + pp[:j]
            
        if j == 40:
            line5.append(40)
            line5 = line5 + pp[:j]
            
            
        if j == 50:
            line6.append(50)
            line6 = line6 + pp[:j]
    
    
    with open( 'a1_3.3.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(line1)   
        writer.writerow(line2) 
        writer.writerow(line3) 
        writer.writerow(line4) 
        writer.writerow(line5) 
        writer.writerow(line6)
        writer.writerow(line7)
        writer.writerow(line8)
      
    print("Class33 done!")  
        
def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    
    print("Doing class34")
    print(i)
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    fold_1 = []
    fold_2 = []
    fold_3 = []
    fold_4 = []
    fold_5 = []
    p_values = []
    #read data and use Kfold to make 5 folds.
    data = np.load(filename)
    data = data['arr_0']
    X, y = data[:,0:-1], data[:,-1]
    kf = KFold(n_splits=5,shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train_list.append(X[train_index])
        X_test_list.append(X[test_index])
        y_train_list.append(y[train_index])
        y_test_list.append(y[test_index])
    
    for k in range(5):
        accuracy_list = []
        X_train = X_train_list[k]
        X_test = X_test_list[k]
        y_train = y_train_list[k]
        y_test = y_test_list[k]
        #1.for clf linear
        clf_linear = SVC(kernel='linear',max_iter=1000)
        clf_linear.fit(X_train, y_train)
        y_pred_linear = clf_linear.predict(X_test)
        matrix_linear = confusion_matrix(y_test, y_pred_linear)
        accuracy_list.append(accuracy(matrix_linear))
        #2.for clf rbf
        clf_rbf = SVC(kernel='rbf',gamma=2,max_iter=1000)
        clf_rbf.fit(X_train, y_train)
        y_pred_rbf = clf_rbf.predict(X_test)
        matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
        accuracy_list.append(accuracy(matrix_rbf))
        #3.for forest
        clf_forest = RandomForestClassifier(n_estimators=10, max_depth=5)
        clf_forest.fit(X_train,y_train)
        y_pred_forest = clf_forest.predict(X_test)
        forest_matrix = confusion_matrix(y_test, y_pred_forest)
        accuracy_list.append(accuracy(forest_matrix))
        #4.for MLP
        clf_mlp = MLPClassifier(alpha=0.05)
        clf_mlp.fit(X_train,y_train)
        y_pred_mlp = clf_mlp.predict(X_test)
        mlp_matrix = confusion_matrix(y_test,y_pred_mlp)
        accuracy_list.append(accuracy(mlp_matrix))
        #5.for AdaBoost
        clf_ada = AdaBoostClassifier()
        clf_ada.fit(X_train,y_train)
        y_pred_ada = clf_ada.predict(X_test)
        ada_matrix = confusion_matrix(y_test,y_pred_ada) 
        accuracy_list.append(accuracy(ada_matrix))
        
        if k == 0:
            fold_1 = accuracy_list
        
        if k == 1:
            fold_2 = accuracy_list
            
        if k == 2:
            fold_3 = accuracy_list
        
        if k == 3:
            fold_4 = accuracy_list
        
        if k == 4:
            fold_5 = accuracy_list
            
    matrix = np.array([fold_1,fold_2,fold_3,fold_4,fold_5])
    a=matrix[:,i-1]
    for k in range(5):
        if k != i-1 :
            b = matrix[:,k]
            S = stats.ttest_rel(a, b)
            p_values.append(S.pvalue)
    
            
            
    with open( 'a1_3.4.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(fold_1)   
        writer.writerow(fold_2) 
        writer.writerow(fold_3) 
        writer.writerow(fold_4) 
        writer.writerow(fold_5) 
        writer.writerow(p_values) 

        
    print("Class34 done!") 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    #TODO : complete each classification experiment, in sequence.
    X_train, X_test, y_train, y_test,iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test,iBest)
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.input, iBest)
    
