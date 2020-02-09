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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

from mtcnn import MTCNN
detector = MTCNN()

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


def preprocess_image(img):
    imag=cv2.imread(img)
    res = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    res=np.expand_dims(res,axis=0)
    return res

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    #euclidean_distance = l2_normalize(euclidean_distance )
    return euclidean_distance


#import tensorflow as tf
model = load_model('facenet/facenet_keras.h5')
model.summary()
print(model.inputs)
print(model.outputs)

model.load_weights("facenet/facenet_keras_weights.h5")

'''
def createModel():
    
    inputShape=(128,)
    inputs      = Input(shape=inputShape)
    #x=Reshape((128,1))(inputs)
    x = Dense(128,activation="relu")(inputs)
    #x=Conv1D(128,kernel_size=(8,),activation="relu",padding="same")(x)
    #x=Conv1D(64,kernel_size=(4,),activation="relu",padding="same")(x)
    #x=AveragePooling1D(4)(x)
    #x = Flatten()(x)
    x = Dense(64,activation="relu")(x)
    x = Dense(32,activation="relu")(x)
    x = Dense(20,activation="relu")(x)

    outputs0 = Dense(20,activation="relu")(x)
    outputs1 = Dense(20,activation="relu")(x)
    outputs2 = Dense(20,activation="relu")(x)
    outputs3 = Dense(20,activation="relu")(x)
    
    outputs0 = Dense(2,activation="softmax")(outputs0)
    outputs1 = Dense(2,activation="softmax")(outputs1)
    outputs2 = Dense(2,activation="softmax")(outputs2)
    outputs3 = Dense(2,activation="softmax")(outputs3)
    
    model       = Model(inputs=inputs,outputs=[outputs0,outputs1,outputs2,outputs3])       
    #model       = Model(inputs=[inputs0,inputs1,inputs2,inputs3,inputs4,inputs5,inputs6,inputs7],outputs=outputs)       
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizers.Adam() ,
                metrics=['accuracy'])

    return model
'''

#model2=createModel()
model2=load_model('facenet_network_model.hdf5')
model2.summary()
modelname="facenet_network"
model2.load_weights(modelname + ".hdf5")

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

filename="voting0.sav"
model15=pickle.load(open(filename,'rb'))
filename="voting1.sav"
model16=pickle.load(open(filename,'rb'))
filename="voting2.sav"
model17=pickle.load(open(filename,'rb'))
filename="voting3.sav"
model18=pickle.load(open(filename,'rb'))


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

prev=0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #cv2.imwrite("frame.jpg",frame)
    #sleep(0.25)

    '''
    movement=0    
    
    if prev==0:
        #prev_hist=histr
        prev_frame=frame
        prev=1
        sleep=(0.25)
        continue
    else:
        color = ('b','g','r')
        histr=[]
        for i,col in enumerate(color):
            histr = moving_average( cv2.calcHist([frame],[i],None,[256],[0,256]) )
            #print(histr)
            #histr2 = moving_average( cv2.calcHist([prev_frame],[i],None,[256],[0,256]) )
            #print(histr2)     
            #print(histr-histr2)
            #raise
            #histr3 = histr-histr2 #moving_average( cv2.calcHist([frame-prev_frame],[i],None,[256],[0,256]) )
            #histr3/=histr+1
            #print(histr3)
            #raise
            #plt.plot(histr,color = col)
            #plt.plot(histr2,color = col)
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            plt.show()

        #plt.plot(histr,color = col)
        #plt.plot(histr2,color = col)
        #plt.plot(histr3,color = col)
        #plt.plot(prev_hist,color = col)
        #plt.plot(histr-prev_hist,color = col)
        #plt.xlim([0,256])
        #plt.show()
        #prev_hist=histr
        prev_frame=frame
        sleep=(1)
    '''
    '''    
    negative=frame-prev_frame
    negative=np.where(negative>240,0,negative)
    negative=np.where(negative<15,0,negative)
    '''
    
    
    #grayplt(negative)
    #grayplt(frame)
    #grayplt(prev_frame)
    prev_frame=frame
    #print( np.size(negative))
    #print( np.sum(negative>120)  )
    #sleep(1
    #print(frame)
    #print(prev_frame)
    #raise


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        resized = cv2.resize(frame[y:y+h,x:x+w], (160,160), interpolation = cv2.INTER_AREA)
        grayplt(resized/255)
        
        result = detector.detect_faces(resized)
        print(result)
        if result==[]: continue
        
        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        bounding_box = result[0]['box']
        
        resized=resized[ bounding_box[1]:bounding_box[1]+bounding_box[3] , bounding_box[0]:bounding_box[0]+bounding_box[2] ]
        #grayplt(image/255)
        resized = cv2.resize(resized,(160, 160), interpolation = cv2.INTER_CUBIC)
        grayplt(resized/255)
        result = detector.detect_faces(resized)
        print(result)  
        if result==[]: continue
        
        keypoints = result[0]['keypoints']
        while keypoints['right_eye'][1]-keypoints['left_eye'][1]>8:
            img = ndimage.rotate(resized, 2, mode='nearest')
            #print(img.shape)
            trim1=(img.shape[0]-160)/(2)
            trim1=int(trim1)
            trim2=(img.shape[1]-160)/(2)
            trim2=int(trim2)
            resized=img[trim1:trim1+160,trim2:trim2+160]
            print("turned")
            grayplt(resized/255)
            result = detector.detect_faces(resized)
            if result==[]: break
            keypoints = result[0]['keypoints']
        while keypoints['right_eye'][1]-keypoints['left_eye'][1]<-8:
            img = ndimage.rotate(resized, -2, mode='nearest')
            #print(img.shape)
            trim1=(img.shape[0]-160)/(2)
            trim1=int(trim1)
            trim2=(img.shape[1]-160)/(2)
            trim2=int(trim2)
            resized=img[trim1:trim1+160,trim2:trim2+160]
            print("turned")
            grayplt(resized/255)
            result = detector.detect_faces(resized)
            if result==[]: break
            keypoints = result[0]['keypoints']

        #left_eye=image[keypoints['left_eye'][1]-20:keypoints['left_eye'][1]+20, keypoints['left_eye'][0]-20:keypoints['left_eye'][0]+20]
        #grayplt(left_eye/255)
        new_bound=min(keypoints['left_eye'][1],keypoints['right_eye'][1])/3
        new_bound=int(new_bound)
        new_bound1=(160-max(keypoints['mouth_left'][1],keypoints['mouth_right'][1]))/3
        new_bound1=int(new_bound1)
        new_bound2=keypoints['left_eye'][0]/4
        new_bound2=int(new_bound2)
        new_bound3=(160-keypoints['right_eye'][0])/4
        new_bound3=int(new_bound3)
        resized = cv2.resize(resized[new_bound:160-new_bound1,new_bound2:160-new_bound3],(160, 160), interpolation = cv2.INTER_CUBIC)
        
        
        img_temp=resized/255
        '''999
-        ######################
        img_temp=resized/255
        adjusted = adjust_gamma(resized, gamma=1.2)
        #cv2.imshow('frame', adjusted)
        img=adjusted
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
         
        # define range of blue color in HSV
        lower_blue= np.array([0,10,45])
        upper_blue = np.array([55,180,255])
        lower_blue= np.array([0,10,45])
        upper_blue = np.array([180,180,255])
        
            
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        #print(res[res<0.1].shape)
        if res[res<0.1].shape[0]>0.95*(res.shape[0]*res.shape[1]*res.shape[2]):
            continue
        
        imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(255-imgray, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print("contours",contours)
        going=0
        try:
            cnt = contours[4]
        except:
            going=1
            pass
        if going==1: continue
        cv2.drawContours(im2, [cnt], 0, (255,255,255), 3)
        im2=255-im2    
        img_temp2=np.expand_dims(img_temp,axis=0)
        #grayplt(img_temp2[0])
        
        img=np.expand_dims(img,axis=0)/255
        res=np.expand_dims(res,axis=0)/255
        999'''
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)    
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
        #grayplt(res/255)
        
        h, s, v = cv2.split(hsv)
        #print("h")
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
        #grayplt(com/255)
        
        #h[h>20]=180
        #h[h<=20]=0
        
        #grayplt(h/180)
        #print("s")
        l=cv2.resize(s,(160, 160), interpolation = cv2.INTER_CUBIC)
        l[l>185]=255
        l[l<=185]=0      
        com2=cv2.resize(l,(160, 160), interpolation = cv2.INTER_CUBIC)
        #grayplt(com2/255)
        #grayplt(s/255)
        #print("v")
        v[v<100]=0
        v[(v<120)&(v!=0)]-=30
        v[(v<210)&(v!=0)]-=20
        v[v>235]=0
        #grayplt(v/255)
    
        b, g, r = cv2.split(res)
        #print("b")
        #l=cv2.resize(b,(160, 160), interpolation = cv2.INTER_CUBIC)
        b[b>230]=255
        b[b<=230]=0      
        #grayplt(b/255)
        
        #grayplt(b/255)
        #g[g<50]=0
        #g[g>=50]=255
        #print("g")
        g[g<60]=1
        g[g>=60]=254
        g[g==1]=255
        g[g==254]=0
        #grayplt(g/255)
        
        #print("r")
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
        #grayplt(r/255)
        
        #print("combined")
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
        
        im2=fin*255
        #grayplt( fin )
        
        
        #9999
        im2=im2/255
        im3=np.fliplr(im2)
        im4=np.flipud(im2)
        im5=np.fliplr(im4)
        
        for i in range(15):
            im2=ndimage.maximum_filter(im2, size=2)
            im3=ndimage.maximum_filter(im3, size=2)
            im4=ndimage.maximum_filter(im4, size=2)
            im5=ndimage.maximum_filter(im5, size=2)
            
            im2=ndimage.maximum_filter(im2, size=2)
            im3=ndimage.maximum_filter(im3, size=2)
            im4=ndimage.maximum_filter(im4, size=2)
            im5=ndimage.maximum_filter(im5, size=2)
            
            #im2=scipy.ndimage.gaussian_filter(im2, sigma=1.1)
            im2=ndimage.minimum_filter(im2, size=2)
            im2=ndimage.minimum_filter(im3, size=2)
            im2=ndimage.minimum_filter(im4, size=2)
            im2=ndimage.minimum_filter(im5, size=2)
        
        im3=np.fliplr(im3)
        im4=np.flipud(im4)
        im5=np.fliplr(im5)
        im5=np.flipud(im5)
        
        im2=im2*im3*im4*im5
        
        #print(im2[im2<0.1].shape)
        if im2[im2<0.1].shape[0]>0.95*(im2.shape[0]*im2.shape[1]):
            continue
        
        img2 = np.zeros( ( np.array(im2).shape[0], np.array(im2).shape[1], 3 ) )
        img2[:,:,0] = im2 # same value in each channel
        img2[:,:,1] = im2
        img2[:,:,2] = im2
        
        im22=img2*img_temp
        grayplt(im22)
        

        im2=im22*255
        #print(im2.shape)
        
        res5=im2
        #res=np.expand_dims(im2,axis=0)
        
        resized=np.expand_dims(im2,axis=0)
        ######################
        
        #resized=np.expand_dims(resized,axis=0)
        #p1 = 'frame5.jpg'
        #p2 = 'image2/frame3.jpg'
        #p2 = 'image2/frame2.jpg'
         
        #img1_representation = model.predict(preprocess_image(p1))[0,:]
        img2_representation = model.predict(resized) #(preprocess_image(resized))[0,:]
        result2=model3.predict(img2_representation)
        result3=model4.predict(img2_representation)
        result4=model5.predict(img2_representation)
        result5=model6.predict(img2_representation)
        
        result6=model7.predict(img2_representation)
        result7=model8.predict(img2_representation)
        result8=model9.predict(img2_representation)
        result9=model10.predict(img2_representation)
        
        result10=model11.predict(img2_representation)
        result11=model12.predict(img2_representation)
        result12=model13.predict(img2_representation)
        result13=model14.predict(img2_representation)

        result14=model15.predict(img2_representation)
        result15=model16.predict(img2_representation)
        result16=model17.predict(img2_representation)
        result17=model18.predict(img2_representation)
        
        print(img2_representation.shape)
        print("svm")
        print("francis",result2)
        print("Yu Ka",result3)
        print("boonping",result4)
        print("aujunleng",result5)
        
        print("lr")
        print("francis",result6)
        print("Yu Ka",result7)
        print("boonping",result8)
        print("aujunleng",result9)

        print("knn")
        print("francis",result10)
        print("Yu Ka",result11)
        print("boonping",result12)
        print("aujunleng",result13)

        print("voting")
        print("francis",result14)
        print("Yu Ka",result15)
        print("boonping",result16)
        print("aujunleng",result17)
        
        '''
        cosine = findCosineDistance(img1_representation, img2_representation)
        euclidean = findEuclideanDistance(img1_representation, img2_representation)
        
        if cosine <= 0.02:
           print("this is boonping")
        else:
           print("this is not boonping")
        '''
        
        
        prediction=model2.predict(img2_representation)
        #print(np.argmax(prediction[0]))
        print(prediction)
        
        fa=-1
        val=0
        sel=0
        for fac in range(3):
            fa+=1
            if np.argmax(prediction[fac][0])==1:
                if fa==0 and prediction[fac][0][1]>val: 
                    sel=1
                    val=prediction[fac][0][1]

                    
                if fa==1 and prediction[fac][0][1]>val: 
                    sel=2
                    val=prediction[fac][0][1]


                    
                if fa==2 and prediction[fac][0][1]>val: 
                    sel=3
                    val=prediction[fac][0][1]

                if fa==3 and prediction[fac][0][1]>val: 
                    sel=4
                    val=prediction[fac][0][1]


                    
                    
        if val<0.55: sel=0
        
        if sel==0: print("not recognized")
        elif sel==1: print("Francis")
        elif sel==2: print("Yu Ka")
        elif sel==3: print("BoonPing")
        elif sel==4: print("JunLeng")
        
                
        '''
        if np.argmax(prediction[0])==1:
           print("this is boonping")
        else:
           print("this is not boonping")
        ''' 
        

        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sleep(0.5)
        '''
        resized=np.expand_dims(resized,axis=0)/255
        print(resized.shape)
        

        
        predicts_img    = modelGo.predict(resized)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        grayplt(resized[0])
        print(np.argmax(predicts_img[0]))
        print(predicts_img[0])
        
        image = load_img("frame7.jpg")
        resized2=np.expand_dims(image,axis=0)/255
        predicts_img    = modelGo.predict(resized2)
        grayplt(resized2[0])
        print(np.argmax(predicts_img[0]))
        print(predicts_img[0])
        
        cv2.putText(frame,'%s' % np.argmax(predicts_img), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
        '''
        
        '''
        cv2.imwrite("frame.jpg",frame[y:y+h,x:x+w])
        #cv2.imwrite("frame.jpg",frame)
        resized = cv2.resize(frame[y:y+h,x:x+w], (200,200), interpolation = cv2.INTER_AREA)
        cv2.imwrite("frame2.jpg",resized)
        
        #raise
        '''
        

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        '''
        image = load_img("frame2.jpg")
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
        print("[INFO] generating images...")
        imageGen = aug.flow(image, batch_size=1, save_to_dir=".",save_prefix="image5", save_format="jpg")
        i=0
        for image in imageGen:
            print(image)
            i+=1
            if i==100: break
        '''
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
