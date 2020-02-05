# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:33:20 2020

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

def process_csv():
    X0 = loadtxt('img0_merged_representation.csv', delimiter=',')
    X1 = loadtxt('img1_merged_representation.csv', delimiter=',')
    X2 = loadtxt('img2_merged_representation.csv', delimiter=',')
    X3 = loadtxt('img3_merged_representation.csv', delimiter=',')
    X4 = loadtxt('img4_merged_representation.csv', delimiter=',')
    
    
    X5 = loadtxt('img5_merged_representation.csv', delimiter=',')
    X6 = loadtxt('img6_merged_representation.csv', delimiter=',')
    X7 = loadtxt('img7_merged_representation.csv', delimiter=',')
    X8 = loadtxt('img8_merged_representation.csv', delimiter=',')
    X9 = loadtxt('img9_merged_representation.csv', delimiter=',')
    X10 = loadtxt('img10_merged_representation.csv', delimiter=',')
    X11 = loadtxt('img11_merged_representation.csv', delimiter=',')
    X12 = loadtxt('img12_merged_representation.csv', delimiter=',')
    X13 = loadtxt('img13_merged_representation.csv', delimiter=',')
    X14 = loadtxt('img14_merged_representation.csv', delimiter=',')
    X15 = loadtxt('img15_merged_representation.csv', delimiter=',')
    X16 = loadtxt('img16_merged_representation.csv', delimiter=',')
    X17 = loadtxt('img17_merged_representation.csv', delimiter=',')
    X18 = loadtxt('img18_merged_representation.csv', delimiter=',')
    X19 = loadtxt('img19_merged_representation.csv', delimiter=',')
    X20 = loadtxt('img20_merged_representation.csv', delimiter=',')
    X21 = loadtxt('img21_merged_representation.csv', delimiter=',')
    X22 = loadtxt('img22_merged_representation.csv', delimiter=',')
    X23 = loadtxt('img23_merged_representation.csv', delimiter=',')

    Y0=np.append( np.ones(X0.shape[0]), np.ones(X1.shape[0]),axis=0)
    Y0=np.append( Y0, np.ones(X2.shape[0]),axis=0)
    Y0=np.append( Y0, np.ones(X3.shape[0]),axis=0)
    Y0=np.append( Y0, np.ones(X4.shape[0]),axis=0)
    Y0=np.append( Y0, np.zeros(X5.shape[0]),axis=0)
    Y0=np.append( Y0, np.zeros(X6.shape[0]),axis=0)
    Y0=np.append( Y0, np.zeros(X7.shape[0]),axis=0)
    Y0=np.append( Y0, np.zeros(X8.shape[0]),axis=0)
    Y0=np.append( Y0, np.zeros(X9.shape[0]),axis=0)
    Y0=np.append( Y0, np.zeros(X10.shape[0]),axis=0)
    Y0=np.append( Y0, np.zeros(X11.shape[0]),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    Y0=np.append( Y0, np.zeros(50),axis=0)
    
    
    Y1=np.append( np.zeros(X0.shape[0]),  np.zeros(X1.shape[0]),axis=0)
    Y1=np.append( Y1, np.zeros(X2.shape[0]),axis=0)
    Y1=np.append( Y1, np.zeros(X3.shape[0]),axis=0)
    Y1=np.append( Y1, np.zeros(X4.shape[0]),axis=0)
    Y1=np.append( Y1, np.ones(X5.shape[0]),axis=0)
    Y1=np.append( Y1, np.ones(X6.shape[0]),axis=0)
    Y1=np.append( Y1, np.ones(X7.shape[0]),axis=0)
    Y1=np.append( Y1, np.ones(X8.shape[0]),axis=0)
    Y1=np.append( Y1, np.ones(X9.shape[0]),axis=0)
    Y1=np.append( Y1, np.zeros(X10.shape[0]),axis=0)
    Y1=np.append( Y1, np.zeros(X11.shape[0]),axis=0)
    Y1=np.append( Y1, np.zeros(50),axis=0)
    Y1=np.append( Y1, np.zeros(50),axis=0)

    Y1=np.append( Y1, np.zeros(50),axis=0)
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
    Y2=np.append( Y2, np.zeros(X2.shape[0]),axis=0)
    Y2=np.append( Y2, np.zeros(X3.shape[0]),axis=0)
    Y2=np.append( Y2, np.zeros(X4.shape[0]),axis=0)
    
    Y2=np.append( Y2, np.zeros(X5.shape[0]),axis=0)
    Y2=np.append( Y2, np.zeros(X6.shape[0]),axis=0)
    Y2=np.append( Y2, np.zeros(X7.shape[0]),axis=0)
    Y2=np.append( Y2, np.zeros(X8.shape[0]),axis=0)
    Y2=np.append( Y2, np.zeros(X9.shape[0]),axis=0)
    Y2=np.append( Y2, np.ones(X10.shape[0]),axis=0)
    Y2=np.append( Y2, np.zeros(X11.shape[0]),axis=0)
    Y2=np.append( Y2, np.zeros(50),axis=0)
    Y2=np.append( Y2, np.zeros(50),axis=0)

    Y2=np.append( Y2, np.zeros(50),axis=0)
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
    Y3=np.append( Y3, np.zeros(X3.shape[0]),axis=0)
    Y3=np.append( Y3, np.zeros(X4.shape[0]),axis=0)
    
    Y3=np.append( Y3, np.zeros(X5.shape[0]),axis=0)
    Y3=np.append( Y3, np.zeros(X6.shape[0]),axis=0)
    Y3=np.append( Y3, np.zeros(X7.shape[0]),axis=0)
    Y3=np.append( Y3, np.zeros(X8.shape[0]),axis=0)
    Y3=np.append( Y3, np.zeros(X9.shape[0]),axis=0)
    Y3=np.append( Y3, np.zeros(X10.shape[0]),axis=0)
    Y3=np.append( Y3, np.ones(X11.shape[0]),axis=0)
    Y3=np.append( Y3, np.zeros(50),axis=0)
    Y3=np.append( Y3, np.zeros(50),axis=0)

    Y3=np.append( Y3, np.zeros(50),axis=0)
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

    X=np.append(X,X5,axis=0)
    X=np.append(X,X6,axis=0)
    X=np.append(X,X7,axis=0)
    X=np.append(X,X8,axis=0)
    X=np.append(X,X9,axis=0)

    X=np.append(X,X10,axis=0)
    X=np.append(X,X11,axis=0)
    
    
    for i in [0,600,1200,1800,2400]:
        X=np.append(X,X12[i:i+10],axis=0)
        X=np.append(X,X13[i:i+10],axis=0)
        X=np.append(X,X14[i:i+10],axis=0)
        X=np.append(X,X15[i:i+10],axis=0)
        X=np.append(X,X16[i:i+10],axis=0)
        X=np.append(X,X17[i:i+10],axis=0)
        X=np.append(X,X18[i:i+10],axis=0)
        X=np.append(X,X19[i:i+10],axis=0)
        X=np.append(X,X20[i:i+10],axis=0)
        X=np.append(X,X21[i:i+10],axis=0)
        X=np.append(X,X22[i:i+10],axis=0)
        X=np.append(X,X23[i:i+10],axis=0)
        
        
    X_train,X_test,Y_train0,Y_test0,Y_train1,Y_test1,Y_train2,Y_test2,Y_train3,Y_test3 = train_test_split(X,Y0,Y1,Y2,Y3,test_size = 0.1)
    return X_train,X_test,Y_train0,Y_test0,Y_train1,Y_test1,Y_train2,Y_test2,Y_train3,Y_test3