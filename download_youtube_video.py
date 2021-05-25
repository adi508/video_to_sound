# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:20:20 2021

@author: Adi
"""
main_phath =r'D:\github\video_to_sound\video_to_sound'

import cv2
import numpy as np
import os
from pytube import YouTube
import librosa


phath_for_url = r'\data\url_youtube.txt'
phath_for_saving_video = r'\data\raw_video'
phath_for_saving_video_as_numpy = r'\data\matrix_video'


#url_test = 'https://www.youtube.com/watch?v=4vvBAONkYwI&ab_channel=BritneySpearsBritneySpears'
url_test = open(main_phath+phath_for_url,'r')

print(main_phath+phath_for_url)

for index,singal_url in enumerate(url_test): 
    print(index,'singal_url:',singal_url)
    try: 
          
        # object creation using YouTube
        # which was imported in the beginning 
        yt = YouTube(singal_url)
        number_of_sec = yt.length
        print(yt.title)
    except: 
          
        #to handle exception
        print("Connection\\url Error")  
        continue  #try next link
      
     
    #set stream resolution
    my_video = yt.streams.get_highest_resolution()

        
    #Download video
    my_video.download(main_phath+phath_for_saving_video)
    
    #convert mp4 to 3D numpy array [w,h,t] in black and white
    raw_video_phath = main_phath+phath_for_saving_video+'\\'+yt.title+'.mp4'
    cap = cv2.VideoCapture(raw_video_phath)
    number_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('time_of_frame',number_of_sec/number_of_frame)

    print(raw_video_phath)
    #new_name = main_phath+phath_for_saving_video_as_numpy+yt.title
    counter = 0 
    ret,frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    numpy_video = np.expand_dims(grayFrame,axis=-1)
    while(True):   
        # reading from frame 
        ret,frame = cap.read() 
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayframe = np.expand_dims(grayFrame,axis=-1)
        print('in loop',ret)

        if ret: 
            # if video is still left continue creating images 
            name = './data/frame'+str(index)+'_' + str(counter) + '.jpg'
            #print ('Creating...' + name) 
            print('frame',type( grayframe), grayframe.shape)
            # writing the extracted images 
            numpy_video = np.append(numpy_video,grayframe,axis=-1)
            print(numpy_video.shape,type(numpy_video))
            #cv2.imwrite(name,  grayFrame) 

            # increasing counter so that it will 
            # show how many frames are created 
            counter += 1
            if  counter>30:
                break
        else: 
            break

# Release all space and windows once done 
cap.release() 
cv2.destroyAllWindows()
    
    
   
    
    
print('Task Completed!') 











