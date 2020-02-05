import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

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
        ax.imshow(img[:,:,0],cmap='hot',vmin=0,vmax=255)
    else:
        ax.imshow(img,cmap='hot',vmin=0,vmax=255)
   
    plt.show()

#Using Haar Wavelet to capture the face
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

    movement=0
    
    #prev_frame=frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Use HAar Wavelet to get the frame that consists of human face
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # x,y,w,h defines frame captured by Haar Wavelet
    for (x, y, w, h) in faces:
        #for each frame, resized to 160,160
        resized = cv2.resize(frame[y:y+h,x:x+w], (160,160), interpolation = cv2.INTER_AREA)
        #USe MTCNN to detect face
        result = detector.detect_faces(resized)
        #Go to next captured face if the HAar Wavelet is not capturing a face but some other object that has face color/feature
        if result==[]: continue
        #If successful, bounding box is defined by the MTCNN output.
        bounding_box = result[0]['box']
        resized=resized[ bounding_box[1]:bounding_box[1]+bounding_box[3] , bounding_box[0]:bounding_box[0]+bounding_box[2] ]
        #Resize to 160,160 again.
        resized = cv2.resize(resized,(160, 160), interpolation = cv2.INTER_CUBIC)
        #Detect face again.
        result = detector.detect_faces(resized)
        print(result)    
        #If crop the image accidentally, skip
        if result==[]: continue
        
        #save image as frame.jpg
        cv2.imwrite("frame.jpg",resized)
        print("Image updated")
        grayplt(resized)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        #sleep(0.5)
        
        #raise

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        #ret, frame = video_capture.read()
        #cv2.imwrite("frame98.jpg",resized)        
        #image = load_img("frame98.jpg")
        #image = img_to_array(image)
        #image = np.expand_dims(image, axis=0)
        '''
        aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
        print("[INFO] generating images...")
        imageGen = aug.flow(image, batch_size=1, save_to_dir=".",save_prefix="image1", save_format="jpg")
        i=0
        for image in imageGen:
            print(image)
            i+=1
            if i==100: break
        
        break
        '''
        #cv2.imwrite("frame.jpg",resized)
        resized=cv2.imread('frame.jpg')
        grayplt(resized/255)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    #print("P")

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
