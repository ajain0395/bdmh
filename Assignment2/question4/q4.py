# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

#from Pfeature.Pfeature import

posclassfile = open("./Data/q4/225_amp_pos.txt") 

negclassfile = open("./Data/q4/neg_2250.txt") 

train_test_split_ratio = 0.7

#print (posclassfile.readline())

def reports(classifier,train_data,train_labels):
    kf = KFold(n_splits=5)
    kf.get_n_splits(train_data)
    print(kf)
    scores = []
    for train_index, test_index in kf.split(train_data):
        #print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        classifier.fit(X_train,y_train)
        predicted = classifier.predict(X_test)
        scores.append(accuracy_score(predicted,y_test))
    scores = np.array(scores)
    print ("Average Accuracy K Fold: ",scores.mean())
    train_data_len = len(train_data)
    chunksize = int(train_data_len*train_test_split_ratio)
    
    train_x = train_data[0:chunksize]
    train_y = train_labels[0:chunksize]

    test_x = train_data[chunksize:train_data_len]
    test_y = train_labels[chunksize:train_data_len]
    
    classifier.fit(train_x,train_y)
    predicted = classifier.predict(test_x)
    print ("Test Data Results:")
    print ("Test Accuracy: ",accuracy_score(predicted,test_y))
    X = classification_report(test_y,predicted,output_dict=True)
    #print (X.keys())
    print ("Sensitivity: ", X['1']['recall'])
    print ("Specificity: ", X['0']['recall'])
    print ("MCC: ",mcc(test_y,predicted))
    print ("")
    


def getaac(zz):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    #print("A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,")
    empty = []
    for i in std:
        count = 0
        for k in zz:
            temp1 = k
            if temp1 == i:
                count += 1
            composition = (count/len(zz))*100
            #print("%.2f"%composition, end = ",")
        empty.append(composition)
    return empty

train_x = []
train_y = []
for i in posclassfile.readlines():
    train_x.append(getaac(i.strip()))
    train_y.append(1)
for i in negclassfile.readlines():
    train_y.append(0)
    train_x.append(getaac(i.strip()))
    
train_x,train_y = shuffle(np.array(train_x),np.array(train_y))
randomforest = RandomForestClassifier(n_estimators=100)
svm = SVC(gamma='auto')
ann = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

print("##########################################")
print("SVM")
reports(svm,train_x,train_y)

print("##########################################")
print("ANN")
reports(ann,train_x,train_y)

print("##########################################")
print("RANDOM FOREST")
reports(randomforest,train_x,train_y)

