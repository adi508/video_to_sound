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
import re

def fix_string(st_in):
    # only lower case
    st0 = st_in.lower() #casefold() ?
    st1 = re.sub("[^0-9a-zA-Z,' ']", "",st0)
    no_good_string =[' hd','hd ','official','video']
    for bad in no_good_string:
        st1 =st1.replace(bad,'')
    st2 =st1
    # remove all space long then 1
    while '  ' in st2:
        st2 = st2.replace("  ", " ")
    
    if len(st2)<2:  #not english
        st22=st0
        st22 = st22.replace("-", "")
        st22 = st22.replace(":", "")
        st22 = st22.replace("@", "")
        st22 = st22.replace("#", "")
        while '  ' in st22:
            st22 = st22.replace("  ", " ")
        st2 = st22
        
    while st2[-1]==' ':
        st2=st2[:-1]
    
    # replace space with '_'
    st3 = st2.replace("_", "") # to prevent double '_'
    st3 = st3.replace(" ", "_")

    return st3
    
phath_for_url = r'\data\url_youtube.txt'
phath_for_saving_video = r'\data\raw_video'
phath_for_saving_video_as_numpy = r'\data\matrix_video'
phath_for_csv_index_data =r'\data\index_all_data.csv'

#url_test = 'https://www.youtube.com/watch?v=4vvBAONkYwI&ab_channel=BritneySpearsBritneySpears'
url_test = open(main_phath+phath_for_url,'r')

print(main_phath+phath_for_url)
all_vide_dic = {}
download_flag = False
dim =(128,72)
for index,singal_url in enumerate(url_test):
    print('-------------------------',index,'---------new video-------')
    print('singal_url:',singal_url)
    try:   
        # object creation using YouTube
        yt = YouTube(singal_url)
        number_of_sec = yt.length
        print(yt.title)
    except:         
        print("Connection\\url Error")  
        continue  #try next link
      
    if download_flag:
        #set stream resolution
        my_video = yt.streams.get_highest_resolution()
    
        #Download video
        my_video.download(main_phath+phath_for_saving_video)
    
    #convert mp4 to 3D numpy array [w,h,t] in black and white
    raw_video_phath = main_phath+phath_for_saving_video+'\\'+yt.title+'.mp4'
    print("raw_video_phath",raw_video_phath)
    cap = cv2.VideoCapture(raw_video_phath)
    print('cap:',type(cap))
    number_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('number_of_sec:',number_of_sec)
    print('number_of_frame:',number_of_frame)
    try:
        print('time_of_frame',number_of_sec/number_of_frame)
    except:
        print('erorrrr')
        print('number of frame is zero')

    print(raw_video_phath)
    counter = 0 
    ret,frame = cap.read()
    print(type(frame),'ret:',ret)
    if not ret:
        print('erorr reading cap- move to next url')
        continue
    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayFrame, dim, interpolation = cv2.INTER_AREA)
    numpy_video = np.expand_dims(resized,axis=0)
   
    while(True):   
        # reading from frame 
        ret,frame = cap.read() 
        if ret and  counter<100 :#and counter<50:    
                        
            #print(numpy_video.shape,type(numpy_video))
            # 

            #  counter for tests
            counter += 1
            
            # covert frame to gray scale and down-size  it
            grayframe0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayframe1 = cv2.resize(grayframe0, dim, interpolation = cv2.INTER_AREA)
            # add dim in the first axis so it can be append to all frames in one 3d array
            grayframe = np.expand_dims(grayframe1,axis=0)
            numpy_video = np.append(numpy_video,grayframe,axis=0)
            
            cv2.imshow('frame', grayframe1)
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            
        else:
            print(numpy_video.shape,number_of_sec/(len(numpy_video)-1),number_of_sec,(len(numpy_video)-1))
            numpy_video[0,0,0] = number_of_sec/(len(numpy_video)-1)
            file_name =  main_phath+ phath_for_saving_video_as_numpy+'\\'+yt.title+'.npy'
            #np.save(file_name,  numpy_video)
            cap.release()
            cv2.destroyAllWindows()
            break

# Release all space and windows once done 
cap.release() 
cv2.destroyAllWindows()
    
    
   
    
    
print('Task Completed!') 











