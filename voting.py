# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:30:47 2020

@author: boonping
"""

import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
from matplotlib import pyplot as plt

'''
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
'''
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import add,Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import IPython
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from numpy import savetxt,loadtxt
#savetxt('data.csv', data, delimiter=',')
#data = loadtxt('data.csv', delimiter=',')
import gc
from skimage.transform import resize

import pickle
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import train_test_split
import process_csv

'''
model = load_model('facenet/facenet_keras.h5')
model.summary()
print(model.inputs)
print(model.outputs)

model.load_weights("facenet/facenet_keras_weights.h5")
'''
'''
model2=load_model('facenet_network_model.hdf5')
model2.summary()
modelname="facenet_network"
model2.load_weights(modelname + ".hdf5")
'''
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

'''
filename="svm0.sav"
model3=pickle.load(open(filename,'rb'))
filename="svm1.sav"
model4=pickle.load(open(filename,'rb'))
filename="svm2.sav"
model5=pickle.load(open(filename,'rb'))
filename="svm3.sav"
model6=pickle.load(open(filename,'rb'))
'''
filename="MLP0.sav"
model3=pickle.load(open(filename,'rb'))
filename="MLP1.sav"
model4=pickle.load(open(filename,'rb'))
filename="MLP2.sav"
model5=pickle.load(open(filename,'rb'))
filename="MLP3.sav"
model6=pickle.load(open(filename,'rb'))


filename="LR0.sav"
model7=pickle.load(open(filename,'rb'))
filename="LR1.sav"
model8=pickle.load(open(filename,'rb'))
filename="LR2.sav"
model9=pickle.load(open(filename,'rb'))
filename="LR3.sav"
model10=pickle.load(open(filename,'rb'))

filename="KNN0.sav"
model11=pickle.load(open(filename,'rb'))
filename="KNN1.sav"
model12=pickle.load(open(filename,'rb'))
filename="KNN2.sav"
model13=pickle.load(open(filename,'rb'))
filename="KNN3.sav"
model14=pickle.load(open(filename,'rb'))



eclf1 = VotingClassifier(estimators=[ ('mlp', model3), ('lr', model7), ('knn', model11)], voting='soft', weights=[1,1,2])
eclf1 .fit(X_train,Y_train0)
filename="voting0.sav"
pickle.dump(eclf1,open(filename,'wb'))
eclf1=pickle.load(open(filename,'rb'))
prediction=eclf1.predict(X_test)
CM=confusion_matrix(Y_test0,prediction)
print("voting0")
print(CM)

eclf2 = VotingClassifier(estimators=[ ('mlp', model4), ('lr', model8), ('knn', model12)], voting='soft', weights=[1,1,2])
eclf2 .fit(X_train,Y_train1)
filename="voting1.sav"
pickle.dump(eclf2,open(filename,'wb'))
eclf2=pickle.load(open(filename,'rb'))
prediction=eclf2.predict(X_test)
CM=confusion_matrix(Y_test1,prediction)
print("voting1")
print(CM)

eclf3 = VotingClassifier(estimators=[ ('mlp', model5), ('lr', model9), ('knn', model13)], voting='soft', weights=[1,1,2])
eclf3 .fit(X_train,Y_train2)
filename="voting2.sav"
pickle.dump(eclf3,open(filename,'wb'))
eclf3=pickle.load(open(filename,'rb'))
prediction=eclf3.predict(X_test)
CM=confusion_matrix(Y_test2,prediction)
print("voting2")
print(CM)

eclf4 = VotingClassifier(estimators=[ ('mlp', model6), ('lr', model10), ('knn', model14)], voting='soft', weights=[1,1,2])
eclf4 .fit(X_train,Y_train3)
filename="voting3.sav"
pickle.dump(eclf4,open(filename,'wb'))
eclf4=pickle.load(open(filename,'rb'))
prediction=eclf4.predict(X_test)
CM=confusion_matrix(Y_test3,prediction)
print("voting3")
print(CM)




