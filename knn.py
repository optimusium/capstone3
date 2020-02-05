# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:01:23 2020

@author: boonping
"""

import cv2
import os,sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
import pickle
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
'''
from sklearn.model_selection import train_test_split

from scipy import ndimage
from scipy.ndimage.interpolation import shift
from numpy import savetxt,loadtxt

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import multilabel_confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import process_csv
'''
X0 = loadtxt('img0_merged_representation.csv', delimiter=',')
X1 = loadtxt('img1_merged_representation.csv', delimiter=',')
X2 = loadtxt('img2_merged_representation.csv', delimiter=',')
X3 = loadtxt('img4_merged_representation.csv', delimiter=',')
X4 = loadtxt('img3_merged_representation.csv', delimiter=',')


X5 = loadtxt('img6_merged_representation.csv', delimiter=',')
X6 = loadtxt('img7_merged_representation.csv', delimiter=',')
X7 = loadtxt('img8_merged_representation.csv', delimiter=',')
X8 = loadtxt('img9_merged_representation.csv', delimiter=',')
X9 = loadtxt('img10_merged_representation.csv', delimiter=',')
X10 = loadtxt('img11_merged_representation.csv', delimiter=',')
X11 = loadtxt('img12_merged_representation.csv', delimiter=',')
X12 = loadtxt('img13_merged_representation.csv', delimiter=',')
X13 = loadtxt('img14_merged_representation.csv', delimiter=',')


Y0=np.append( np.ones(X0.shape[0]), np.zeros(X1.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X2.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X3.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X4.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)

Y1=np.append( np.zeros(X0.shape[0]),  np.ones(X1.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X2.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X3.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X4.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)

Y2=np.append( np.zeros(X0.shape[0]), np.zeros(X1.shape[0]),axis=0)
Y2=np.append( Y2, np.ones(X2.shape[0]),axis=0)
Y2=np.append( Y2, np.zeros(X3.shape[0]),axis=0)
Y2=np.append( Y2, np.zeros(X4.shape[0]),axis=0)

Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)

Y3=np.append( np.zeros(X0.shape[0]), np.zeros(X1.shape[0]),axis=0)
Y3=np.append( Y3, np.zeros(X2.shape[0]),axis=0)
Y3=np.append( Y3, np.ones(X3.shape[0]),axis=0)
Y3=np.append( Y3, np.zeros(X4.shape[0]),axis=0)

Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)


X=X0
X=np.append(X,X1,axis=0)
X=np.append(X,X2,axis=0)
X=np.append(X,X3,axis=0)
X=np.append(X,X4,axis=0)

for i in [0,600,1200,1800,2400]:
    X=np.append(X,X5[i:i+10],axis=0)
    X=np.append(X,X6[i:i+10],axis=0)
    X=np.append(X,X7[i:i+10],axis=0)
    X=np.append(X,X8[i:i+10],axis=0)
    X=np.append(X,X9[i:i+10],axis=0)
    X=np.append(X,X10[i:i+10],axis=0)
    X=np.append(X,X11[i:i+10],axis=0)
    X=np.append(X,X12[i:i+10],axis=0)
    X=np.append(X,X13[i:i+10],axis=0)

print(X.shape)
print(Y3.shape)
#raise
X_train,X_test,Y_train0,Y_test0,Y_train1,Y_test1,Y_train2,Y_test2,Y_train3,Y_test3 = train_test_split(X,Y0,Y1,Y2,Y3,test_size = 0.1)
'''
X_train,X_test,Y_train0,Y_test0,Y_train1,Y_test1,Y_train2,Y_test2,Y_train3,Y_test3 = process_csv.process_csv()

print("KNN0")
#KNN = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=0)
KNN=KNeighborsClassifier(n_neighbors=5) #LogisticRegression(random_state=0, C=1.0)
KNN.fit(X_train,Y_train0)
filename="KNN0.sav"
pickle.dump(KNN,open(filename,'wb'))
KNN=pickle.load(open(filename,'rb'))
prediction=KNN.predict(X_test)
CM=confusion_matrix(Y_test0,prediction)
print(CM)

print("KNN1")
KNN1=KNeighborsClassifier(n_neighbors=5) #LogisticRegression(random_state=0, C=1.0)
#KNN1 = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=5)
KNN1.fit(X_train,Y_train1)
filename="KNN1.sav"
pickle.dump(KNN1,open(filename,'wb'))
KNN1=pickle.load(open(filename,'rb'))
prediction=KNN1.predict(X_test)
CM=confusion_matrix(Y_test1,prediction)
print(CM)

print("KNN2")
#KNN2 = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=5)
KNN2=KNeighborsClassifier(n_neighbors=5) #LogisticRegression(random_state=0, C=1.0)
KNN2.fit(X_train,Y_train2)
filename="KNN2.sav"
pickle.dump(KNN2,open(filename,'wb'))
KNN2=pickle.load(open(filename,'rb'))
prediction=KNN2.predict(X_test)
CM=confusion_matrix(Y_test2,prediction)
print(CM)

print("KNN3")
#
KNN3=KNeighborsClassifier(n_neighbors=5) #LogisticRegression(random_state=0, C=1.0)
#KNN3 = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=5)
KNN3.fit(X_train,Y_train3)
filename="KNN3.sav"
pickle.dump(KNN3,open(filename,'wb'))
KNN3=pickle.load(open(filename,'rb'))
prediction=KNN3.predict(X_test)
CM=confusion_matrix(Y_test3,prediction)
print(CM)
