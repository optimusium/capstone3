# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 19:48:49 2019

@author: boonping
"""

import cv2
import os,sys
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
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import IPython
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from numpy import savetxt,loadtxt
#savetxt('data.csv', data, delimiter=',')
#data = loadtxt('data.csv', delimiter=',')
import gc
from skimage.transform import resize

from mtcnn import MTCNN

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def grayplt(img,title=''):
    '''
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(img[:,:,0],cmap='gray',vmin=0,vmax=1)
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=1)
    plt.title(title, fontproperties=prop)
    '''
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    


    # Show the image
    if np.size(img.shape) == 3:
        ax.imshow(img[:,:,0],cmap='hot',vmin=0,vmax=1)
    else:
        ax.imshow(img,cmap='hot',vmin=0,vmax=1)
   
    plt.show()

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)    


detector = MTCNN()
#images = ['frame1.jpg','frame2.jpg','frame3.jpg','frame4.jpg','frame5.jpg','frame6.jpg','frame7.jpg','frame8.jpg','frame9.jpg','frame10.jpg','frame11.jpg','frame12.jpg','frame13.jpg','frame14.jpg','frame15.jpg','frame16.jpg','frame17.jpg','frame18.jpg','frame19.jpg','frame20.jpg','frame21.jpg','frame22.jpg','frame23.jpg','frame24.jpg','frame25.jpg','frame26.jpg']
#images = ['frame1.jpg','frame2.jpg','frame3.jpg','frame4.jpg','frame5.jpg','frame6.jpg','frame7.jpg','frame8.jpg','frame9.jpg','frame10.jpg']
#images = ['frame25.jpg','frame26.jpg']
#images = ['frame27.jpg']
#images = ['frame1.jpg','frame2.jpg','frame3.jpg','frame4.jpg','frame5.jpg']
#images = ['frame6.jpg','frame7.jpg','frame8.jpg','frame9.jpg','frame10.jpg']
images = ['frame11.jpg','frame12.jpg','frame13.jpg','frame14.jpg','frame15.jpg','frame16.jpg','frame17.jpg','frame18.jpg','frame19.jpg','frame20.jpg','frame21.jpg','frame22.jpg','frame23.jpg','frame24.jpg','frame25.jpg','frame26.jpg']

path=".\\img\\"

model = load_model('facenet/facenet_keras.h5')
model.summary()
print(model.inputs)
print(model.outputs)

model.load_weights("facenet/facenet_keras_weights.h5")


#images = ['frame2.jpg']
#p2 = 'image2/frame3.jpg'
#a=np.array([23,12,15])
#print( a[a<16].size )
#raise

#imags=np.array([])
imags=[]

jjjj=-1
jjj=-1 #offset if adding image
for img in images: #def preprocess_image(img):
    jjjj+=1
    if jjjj%6==0:
        jjj+=1
    #imag=cv2.imread(img)
    image = cv2.cvtColor(cv2.imread(path+img), cv2.COLOR_BGR2RGB)
    #image = ndimage.rotate(image, 30, mode='nearest')
    
    result = detector.detect_faces(image)
    print(result)
    
    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    
    '''
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)
    '''
    grayplt(image/255)
    image=image[ bounding_box[1]:bounding_box[1]+bounding_box[3] , bounding_box[0]:bounding_box[0]+bounding_box[2] ]
    #grayplt(image/255)
    image = cv2.resize(image,(160, 160), interpolation = cv2.INTER_CUBIC)
    result = detector.detect_faces(image)
    print(result)    
    
    keypoints = result[0]['keypoints']
    #left_eye=image[keypoints['left_eye'][1]-20:keypoints['left_eye'][1]+20, keypoints['left_eye'][0]-20:keypoints['left_eye'][0]+20]
    #grayplt(left_eye/255)
    new_bound=min(keypoints['left_eye'][1],keypoints['right_eye'][1])/4
    new_bound=int(new_bound)
    new_bound1=(160-max(keypoints['mouth_left'][1],keypoints['mouth_right'][1]))/4
    new_bound1=int(new_bound1)
    new_bound2=keypoints['left_eye'][0]/4
    new_bound2=int(new_bound2)
    new_bound3=(160-keypoints['right_eye'][0])/4
    new_bound3=int(new_bound3)
    image = cv2.resize(image[new_bound:160-new_bound1,new_bound2:160-new_bound3],(160, 160), interpolation = cv2.INTER_CUBIC)
    grayplt(image/255)
    #raise

    '''
    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    '''
    
    #cv2.imwrite("ivan_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #grayplt(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)/255)
    #print(keypoints['left_eye'][0])
    
    
    imag = cv2.resize( cv2.cvtColor(image, cv2.COLOR_RGB2BGR),(160, 160), interpolation = cv2.INTER_CUBIC)
    grayplt(imag/255)
    if imags==[] : #imags.shape[0]==0:
        #print("999")
        imags=[imag]
    else:
        
        imags.append(imag)
        #imags=np.append(imags,imag,axis=0)
        #print("998",imag.shape)
        #print(imags.shape)
    #imag=adjust_gamma(imag, gamma=0.8)
    hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)    
    h, s, v = cv2.split(hsv)
    v[v<20]=0
    v[(v>20)&(v<120)]=v[(v>20)&(v<120)]*1.08
    v[(v>180)&(v<250)]=v[(v>180)&(v<250)]*0.92
    v[v>250]=245
    hsv = cv2.merge((h, s, v))
    imag2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #grayplt(imag2/255)
    #print(imags.shape)
    #imags=np.append(imags,imag2,axis=0)
    imags.append(imag2)

    hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)    
    h, s, v = cv2.split(hsv)
    v[v<20]=0
    v[(v>20)&(v<120)]=v[(v>20)&(v<120)]*0.92
    v[(v>180)&(v<250)]=v[(v>180)&(v<250)]*0.98
    v[v>250]=250
    hsv = cv2.merge((h, s, v))
    imag2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #grayplt(imag2/255)
    #print(imags.shape)
    #imags=np.append(imags,imag2,axis=0)
    imags.append(imag2)

    b, g, r = cv2.split(imag)
    r[r<15]=0
    r[(r>15)&(r<100)]=r[(r>15)&(r<100)]*1.1
    r[(r>180)&(r<250)]=r[(r>180)&(r<250)]*0.9
    r[r>250]=250

    g[g<15]=0
    g[(g>15)&(g<100)]=g[(g>15)&(g<100)]*1.1
    g[(g>180)&(g<250)]=g[(g>180)&(g<250)]*0.9
    g[g>250]=250

    imag2 = cv2.merge((b, g, r))
    #grayplt(imag2/255)
    #print(imags.shape)
    #imags=np.append(imags,imag2,axis=0)
    imags.append(imag2)
    
    imag2=adjust_gamma(imag, gamma=1.2)
    imags.append(imag2)
    imag2=adjust_gamma(imag, gamma=0.8)
    imags.append(imag2)
    
    '''
    result = detector.detect_faces(imag)
    print(result)    
    
    keypoints = result[0]['keypoints']
    left_eye=imag[keypoints['left_eye'][1]-20:keypoints['left_eye'][1]+20, keypoints['left_eye'][0]-20:keypoints['left_eye'][0]+20]
    hsv = cv2.cvtColor(left_eye, cv2.COLOR_BGR2HSV)    
    h, s, v = cv2.split(hsv)
    v[v<20]=0
    v[(v>20)&(v<120)]=v[(v>20)&(v<120)]*0.20
    v[(v>180)&(v<250)]=v[(v>180)&(v<250)]*0.25
    v[v>250]=250
    hsv = cv2.merge((h, s, v))
    left_eye = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    grayplt(left_eye/255)
    grayplt(imag/255)
    imags.append(imag)
    raise
    '''
     
#imags=imags.reshape(int(imags.shape[0]/160),160,160,3)
#for imag in imags:
#    grayplt(imag/255)
#raise
    
    
os.popen("del *merged_representation_fast*")    
    
jjj=-1
jjjj=-1
for imag in imags: #def preprocess_image(img):
    jjjj+=1
    if jjjj%6==0:
        jjj+=1
    #imag=cv2.imread(img)
    imag = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    
    hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)    
    h, s, v = cv2.split(hsv)
    
    lim = 250
    v[v > lim] = v[v > lim]*0.95
    lim = 220
    v[v > lim] = v[v > lim]*0.95
    lim = 200
    v[v > lim] = v[v > lim]*0.95
    lim = 150
    v[v > lim] = v[v > lim]*0.95
    lim=20
    v[v < lim] = 0

    hsv = cv2.merge((h, s, v))        
    #print (hsv[30][80])
    #print (hsv[80][30])
    
    # define range of blue color in HSV
    res=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #res=hsv
    
    
    '''
    if 1:
        lower_blue= np.array([0,10,45])
        upper_blue = np.array([40,110,255])

        
            
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(imag,imag, mask= mask)
    '''
    
    
    #res = cv2.resize(res,(160, 160), interpolation = cv2.INTER_CUBIC)
    #imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    #hist_item = cv2.calcHist([imgray],[0],None,[256],[0,256])
    #plt.plot(hist_item,color = 'b')
    grayplt(res/255)
    
    h, s, v = cv2.split(hsv)
    print("h")
    #l=cv2.equalizeHist(l)
    l=cv2.resize(h,(160, 160), interpolation = cv2.INTER_CUBIC)
    l=l[0:160,0:120]
    l[l>75]=179
    l[l<=75]=0
    if l[l==0].shape[0] > l[l==179].shape[0]:
        l[l==0]=255
        l[l==179]=0
    else:
        l[l==179]=255
    l[l==0]=100
        
    
    #grayplt(l/180)
    
    ri=cv2.resize(h,(160, 160), interpolation = cv2.INTER_CUBIC)
    ri=ri[0:160,40:160]
    ri[ri>75]=179
    ri[ri<=75]=0
    if ri[ri==0].shape[0] > ri[ri==179].shape[0]:
        ri[ri==0]=255
        ri[ri==179]=0
    else:
        ri[ri==179]=255
    ri[ri==0]=100
    
    #grayplt(ri/180)
    
    com=np.append(l[0:160,0:120],ri[0:160,80:120],axis=1)
    grayplt(com/255)
    
    #h[h>20]=180
    #h[h<=20]=0
    
    #grayplt(h/180)
    print("s")
    l=cv2.resize(s,(160, 160), interpolation = cv2.INTER_CUBIC)
    l[l>185]=255
    l[l<=185]=0      
    com2=cv2.resize(l,(160, 160), interpolation = cv2.INTER_CUBIC)
    grayplt(com2/255)
    #grayplt(s/255)
    print("v")
    v[v<100]=0
    v[(v<120)&(v!=0)]-=30
    v[(v<210)&(v!=0)]-=20
    v[v>235]=0
    grayplt(v/255)

    b, g, r = cv2.split(res)
    print("b")
    #l=cv2.resize(b,(160, 160), interpolation = cv2.INTER_CUBIC)
    b[b>230]=255
    b[b<=230]=0      
    grayplt(b/255)
    
    #grayplt(b/255)
    #g[g<50]=0
    #g[g>=50]=255
    print("g")
    g[g<60]=1
    g[g>=60]=254
    g[g==1]=255
    g[g==254]=0
    grayplt(g/255)
    
    print("r")
    '''
    r=np.where((r>=0)&(r<25),0,r)
    r=np.where((r>=25)&(r<50),25,r)
    r=np.where((r>=50)&(r<75),50,r)
    r=np.where((r>=75)&(r<100),75,r)
    r=np.where((r>=100)&(r<125),100,r)
    r=np.where((r>=125)&(r<150),125,r)
    r=np.where((r>=150)&(r<175),150,r)
    r=np.where((r>=175)&(r<200),175,r)
    r=np.where((r>=200)&(r<250),200,r)
    r=np.where((r>=250)&(r<256),255,r)
    '''
    r[r<40]=1
    r[r>=40]=254
    r[r==1]=255
    r[r==254]=0
    grayplt(r/255)
    
    print("combined")
    fin=(com/255)*(v/255)-(b/255)-(com2/255)-(r/255)-(g/255)
    fin[fin<0.02]=0
    l=fin[0:160,0:10]
    l[l<0.33]=0
    
    '''
    l1=fin[0:160,10:15]
    l1[l1<0.25]=0


    l2=fin[0:160,150:160]
    l2[l2<0.33]=0

    l3=fin[0:160,145:150]
    l3[l3<0.25]=0

    l4=fin[0:25,0:25]
    l4[l4<0.8]=0

    l5=fin[0:25,135:160]
    l5[l5<0.8]=0

    l6=fin[0:25,135:160]
    l6[l6<0.8]=0

    l7=fin[135:160,0:25]
    l7[l7<0.8]=0

    l7=fin[100:140,140:160]
    l7[l7<0.8]-=0.15

    l8=fin[100:140,0:20]
    l8[l8<0.8]-=0.15

    l9=fin[150:160,0:160]
    l9[l9<0.3]-=0.15
    l9[l9<0.5]-=0.12

    l10=fin[0:10,0:160]
    l10[l10<0.3]-=0.15
    l10[l10<0.5]-=0.12
    '''
    
    #fin[fin<0.6]-=0.1
    #fin=fin*fin
    fin[(fin>0.2)&(fin<0.3)]+=0.3
    fin[fin>0.3]+=0.3
    fin[fin>0.9]=1
    
    fin[fin>0.12]=1
    fin[fin<=0.12]=0
    
    fin=fin*255
    grayplt( fin/255 )
    
    
    if 1:
        
        im2=fin/255
        im3=np.fliplr(im2)
        im4=np.flipud(im2)
        im5=np.fliplr(im4)
        #print(im2[im2<0.1].size,160*160*3*0.95)
        #print(im2[im2>0.1].size,160*160*3*0.95)
        
        for i in range(15):
            im2=ndimage.maximum_filter(im2, size=2)
            im3=ndimage.maximum_filter(im3, size=2)
            im4=ndimage.maximum_filter(im4, size=2)
            im5=ndimage.maximum_filter(im5, size=2)
            
            im2=ndimage.maximum_filter(im2, size=2)
            im3=ndimage.maximum_filter(im3, size=2)
            im4=ndimage.maximum_filter(im4, size=2)
            im5=ndimage.maximum_filter(im5, size=2)

        for i in range(3):
            im2=ndimage.minimum_filter(im2, size=2)
            im3=ndimage.minimum_filter(im3, size=2)
            im4=ndimage.minimum_filter(im4, size=2)
            im5=ndimage.minimum_filter(im5, size=2)


            #im2=scipy.ndimage.gaussian_filter(im2, sigma=1.1)
            #im2=ndimage.minimum_filter(im2, size=2)
            #im3=ndimage.minimum_filter(im3, size=2)
            #im4=ndimage.minimum_filter(im4, size=2)
            #im5=ndimage.minimum_filter(im5, size=2)
            #print(im2[im2<0.1].size,160*160*3*0.95)
            #print(im2[im2>0.1].size,160*160*3*0.95)
        
        im3=np.fliplr(im3)
        im4=np.flipud(im4)
        im5=np.fliplr(im5)
        im5=np.flipud(im5)
        #grayplt(im2)
        #grayplt(im3)
        #grayplt(im4)
        #grayplt(im5)
        
        #raise
        im2=im2*im3*im4*im5
        print(im2[im2<0.1].size,160*160*3*0.95)
        print(im2[im2>0.1].size,160*160*3*0.95)

        
        img2 = np.zeros( ( np.array(im2).shape[0], np.array(im2).shape[1], 3 ) )
        img2[:,:,0] = im2 # same value in each channel
        img2[:,:,1] = im2
        img2[:,:,2] = im2
        
        im22=img2
        grayplt(im22*imag/255)
        res5=im22*imag
        im22=im22*imag
        #print(res5.shape)\
        #res5=resize(res5,(160,160))
        #hsv=cv2.cvtColor(res5, cv2.COLOR_BGR2HSV)
        #print(hsv.shape)
        #raise
        #grayplt(res5/255)
        
        training=np.array([])
        res=np.expand_dims(im22,axis=0)
        training=np.append(training,res)
        #print(res5.shape)
        #raise
    
        iii=0
        for sc in range(140,160,10):
            #print("999:",sc)
            #res7 = cv2.resize(res5,(sc, sc), interpolation = cv2.INTER_CUBIC)
            res7=resize(res5,(sc,sc))
            sc1=160-sc
            sc1/=2
            sc1=int(sc1)
            sc2=80-sc1
            #print(sc1)
            res1=np.zeros((sc1,sc1,3))
            res2=np.zeros((160,sc1,3)) #np.concatenate((res1,res1,res1,res1))
            res3=np.zeros((sc1,sc2*2,3)) #np.concatenate((res1,res1),axis=1)
            #print(res.shape)
            #print(res2.shape)
            #print(res3.shape)  
            #print(res5.shape) 
            res4=np.concatenate((res3,res7,res3))
            res6=np.concatenate((res2,res4,res2),axis=1)
        
            #training.append(res.tolist())
            training=np.append(training,res6)
            #grayplt(res6/255)
            ##############
            
            
            
            for ang in [-45,-38,-30,-20,-10,0,10, 20,30,38,45]:
                img = ndimage.rotate(res6, ang, mode='nearest')
                #print(img.shape)
                grayplt(img/255)
                trim1=(img.shape[0]-160)/(2)
                trim1=int(trim1)
                trim2=(img.shape[1]-160)/(2)
                trim2=int(trim2)
                res1=img[trim1:trim1+160,trim2:trim2+160]
                training=np.append(training,res1)  
                grayplt(res1/255)
                #raise
                
                
                shi=20 #int( 30-(sc-80)/2 )
                for sh in [-20,-10,0,10,20]: #range(-shi,shi,10):
                    for sh2 in [-20,-10,0,10,20]: #range(-shi,shi,10):
                        if sh==0 and sh2==0: continue
                        res9 = np.roll(res1, sh, axis=0)
                        if sh<0:
                            res9[160+sh:160,0:160]=0
                        if sh>0:
                            res9[0:sh,0:160]=0
                        res9 = np.roll(res9, sh2, axis=1)
                        if sh2<0:
                            res9[0:160,160+sh2:160]=0
                        if sh2>0:
                            res9[0:160,0:sh2]=0
                        print(sh,sh2)
                        grayplt(res9/255)
                        
                        training=np.append(training,res9)
                #raise
                
                
                       
            print("shape:",training.shape)
            training=training.reshape( int(training.shape[0]/76800),160,160,3)
            
            img1_representation = model.predict(training)
            #savetxt('img%i_representation_%s_%s.csv' % (jjj,iii,sc), img1_representation, delimiter=',')
            with open('img%i_merged_representation_fast.csv' % (jjj), "ab") as f:
                savetxt(f, img1_representation, delimiter=',')
            
            training=np.array([])
            #res5=im2
            #res=np.expand_dims(im2,axis=0)
            training=np.append(training,res)
        iii=1
        for sc in range(180,250,10):
            
            #res1 = cv2.resize(res5,(sc, sc), interpolation = cv2.INTER_CUBIC)
            res1=resize(res5,(sc,sc))
            sc1=(sc-160)/2
            sc1=int(sc1)
            res1=res1[sc1:sc1+160,sc1:sc1+160]
            #print("998",sc)
            grayplt(res1/255)
        
            #training.append(res.tolist())
            training=np.append(training,res1)
            for ang in [-45,-38,-30,-15,0,15,30,38,45]:
                img = ndimage.rotate(res1, ang, mode='nearest')
                #print(img.shape)
                trim1=(img.shape[0]-160)/(2)
                trim1=int(trim1)
                trim2=(img.shape[1]-160)/(2)
                trim2=int(trim2)
                res2=img[trim1:trim1+160,trim2:trim2+160]
                training=np.append(training,res2) 
                #grayplt(res2/255)
                
                if ang<-15: continue
                if ang>15: continue
                shi=30 #int( 30-(sc-80)/2 )
                for sh in [-20,-10,0,10,20]: #range(-shi,shi,10):
                    for sh2 in [-20,-10,0,10,20]: #range(-shi,shi,10):
                        if sh==0 and sh2==0: continue
                        res9 = np.roll(res2, sh, axis=0)
                        if sh<0:
                            res9[160+sh:160,0:160]=0
                        if sh>0:
                            res9[0:sh,0:160]=0                        
                        res9 = np.roll(res9, sh2, axis=1)
                        if sh2<0:
                            res9[0:160,160+sh2:160]=0
                        if sh2>0:
                            res9[0:160,0:sh2]=0
                        training=np.append(training,res9)
                        print(sh,sh2)
                        grayplt(res9/255)

                
    
            print("shape:",training.shape)
            training=training.reshape( int(training.shape[0]/76800),160,160,3)
            
            img1_representation = model.predict(training)
            #savetxt('img%i_representation_%s_%s.csv' % (jjj,iii,sc), img1_representation, delimiter=',')
            with open('img%i_merged_representation_fast.csv' % (jjj), "ab") as ff:
                savetxt(ff, img1_representation, delimiter=',')
            
            training=np.array([])
            #res5=im2
            #res=np.expand_dims(im2,axis=0)
            training=np.append(training,res)


    
    '''
    imgray=cv2.equalizeHist(imgray)
    
    imgray = np.where(imgray<30,0,imgray)
    imgray[imgray>250]=255
    #imgray = np.where(imgray<40 & imgray>20,255,imgray)
    res = cv2.GaussianBlur(imgray,(3,3),0)
    imgray=cv2.equalizeHist(imgray)
    
    imgray_ori = cv2.GaussianBlur(imgray,(3,3),0)
    imgray=cv2.equalizeHist(imgray_ori)
    imgray=cv2.addWeighted(imgray, 0.5, imgray_ori, 0.5, 0, imgray)
    grayplt(imgray/255)
    
      
    imgray_ori = cv2.GaussianBlur(imgray,(3,3),0)
    imgray=cv2.equalizeHist(imgray_ori)
    imgray=cv2.addWeighted(imgray, 0.5, imgray_ori, 0.5, 0, imgray)
    grayplt(imgray/255)
    
    for itime in range(5):      
        imgray_ori = cv2.GaussianBlur(imgray,(3,3),0)
        imgray=cv2.equalizeHist(imgray_ori)
        imgray=cv2.addWeighted(imgray, 0.5, imgray_ori, 0.5, 0, imgray)
        imgray = np.where(imgray<20,0,imgray)
        imgray[imgray>250]=255
        imgray[imgray<30]=0
    '''    

    #imgray[imgray>30]=255
    #grayplt(imgray/255)
    #hist_item = cv2.calcHist([imgray],[0],None,[8],[0,256])
    #print(hist_item)
    #plt.plot(hist_item,color = 'b')
    
    '''
    laplacian = cv2.Laplacian(imgray,cv2.CV_32F)
    laplacian = np.where(laplacian<30,0,laplacian)
    laplacian = np.where(laplacian>30,255,laplacian)
    '''

