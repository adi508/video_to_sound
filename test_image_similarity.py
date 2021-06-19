# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:52:44 2021

@author: Adi
"""
import cv2
import numpy as np
import os
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import re
from moviepy.editor import *
import subprocess
import pandas as pd
import time
import image_similarity
import sound_similarity


main_phath =r'D:\github\video_to_sound\video_to_sound'

# path for file
phath_for_url = main_phath+r'\data\url_youtube.txt'
phath_to_index_data = main_phath+r'\data\index_video_data.csv'

# upload index data
data_index = pd.read_csv(phath_to_index_data).set_index('name')

# path for folder
phath_for_mp4_video = main_phath+r'\data\raw_video'
phath_for_mp3_sound = main_phath+r'\data\mp3_sound'
phath_for_wav_sound = main_phath+r'\data\wav_sound'
phath_for_np_sound = main_phath+r'\data\np_sound'
phath_for_np_video = main_phath+r'\data\np_video'

def test_img1():
    d=3
    dim_out = (128*d,72*d)
    
    video_np_list_name = os.listdir(phath_for_np_video)
    flag_for =False
    for file_name in video_np_list_name:
        print(file_name)
        if data_index.loc[file_name[:-4]]['error']:
            print(data_index.loc[file_name[:-4]]['error'])
            continue
    
        
        numpy_video = np.load(data_index.loc[file_name[:-4]]['path_video_np'])
        counter = 0
        d_counter = 1#data_index.loc[file_name[:-4]]['frame for sec'].astype(int)
        max_count = data_index.loc[file_name[:-4]]['number of farme'].astype(int)
        img1 = numpy_video[0]
        
        all_dis = np.array([0])
        dis_img0 = 8
        while(counter<max_count):
            #wwtime0 = time.time()
            frame = numpy_video[counter]
            counter += d_counter
            # covert frame to gray scale and down-size  it
            grayframe0 = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayframe1 = cv2.resize(grayframe0,dim_out, interpolation = cv2.INTER_AREA)
            # add dim in the first axis so it can be append to all frames in one 3d array
            grayframe = np.expand_dims(grayframe1,axis=0)
    
            dis_img = image_similarity.similarity(img1,frame,20)
            print(end='.')
            dis_from_mean = np.abs(dis_img-np.mean(all_dis))
            all_dis = np.append(all_dis,dis_from_mean)
            
            if dis_from_mean>3*np.mean(all_dis):
                all_dis = np.array([0])
                #cv2.waitKey(2000)
                grayframe1 = cv2.circle(grayframe1, (50,50), 50,(255, 0, 0) ,40)
                print('')
                print('----now---')
            
            # showe frame in win and exit
            cv2.imshow(file_name[:-4], grayframe1)
            k= cv2.waitKey(1)
            if k == ord('q'):  # next clip
                cv2.destroyAllWindows()
                break
            elif k == ord('w'):  # end script
                cv2.destroyAllWindows()
                flag_for = True
                break
            
            elif k==ord('f'):  # forword - RightKey
                counter+=100
                print('forwordwwwwwwww')
                all_dis = np.array([0])
            else:
               pass

            img1 = frame
            #cv2.waitKey(30)
            
        cv2.destroyAllWindows()
        if flag_for:
            break



#test_img1()














