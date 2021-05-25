# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:57:58 2021

@author: Adi
"""
import cv2
import numpy as np
import os
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt

from moviepy.editor import *
import subprocess
#import ffmpeg

main_phath =r'D:\github\video_to_sound\video_to_sound'

phath_for_url = r'\data\url_youtube.txt'
phath_for_mp4_video = r'\data\raw_video'
phath_for_mp3_sound = r'\data\mp3_sound'
phath_for_wav_sound = r'\data\wav_sound'
phath_for_np_sound = r'\data\np_sound'
phath_for_np_video = r'\data\np_video'




phath_for_saving_video_as_numpy = r'\data\matrix_video'


video_raw_list_name = os.listdir(main_phath+phath_for_mp4_video)
for file_name in video_raw_list_name:
    # step 1  convert sound to np
    file_in_phath = main_phath+phath_for_mp4_video+'\\'+file_name
    sound_mp3_phath =  main_phath+phath_for_mp3_sound+'\\'+file_name[:-4]+'.mp3'
    sound_wav_phath =  main_phath+phath_for_wav_sound+'\\'+file_name[:-4]+'.wav'
    sound_np_phath =  main_phath+phath_for_np_sound+'\\'+file_name[:-4]+'.np'
    
    video_clip = VideoFileClip(file_in_phath)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(sound_mp3_phath)
    
    
    #subprocess.call(['ffmpeg', '-i', sound_mp3_phath,sound_wav_phath])
    
    # Calculate RMS
    print(sound_wav_phath)
    (sig, rate) = librosa.load(sound_wav_phath, sr=22050)
    len_in_sec = librosa.get_duration(y=sig, sr=rate)
    print('len_in_sec',len_in_sec)
    sig = (sig*32767).astype(int)
    sig = sig/32767.0
    print(type(sig),type(rate))
    print(sig.shape,rate)
    d = librosa.stft(sig)  # STFT of y
    print(type(d),d.shape)
    s_db = librosa.amplitude_to_db(np.abs(d), ref=np.max)
    print(type(s_db),s_db.shape)
    counter = 0
    f_0 = 0
    for f_temp in s_db.T:
        
        frequency = np.arange(len(f_temp))
        # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)       
        f_temp = f_temp+80
        # ax1.bar(frequency,f_temp)
        # ax2.bar(frequency,f_temp-f_0)
        # ax1.set_title(counter)
        fig,ax = plt.subplots(1, 1, sharey=True)
        ax.bar(frequency,f_temp-f_0)
        ax.set_title(counter)
        plt.pause(0.05)        
        counter += 1
        f_0 = f_temp
        if  counter>100:
            break
        

    
   
    
    
    # counter = 0 
    # ret,frame = cap.read()
    # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # numpy_video = np.expand_dims(grayFrame,axis=-1)
    # while(True):   
    #     # reading from frame 
    #     ret,frame = cap.read() 
    #     grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     grayframe = np.expand_dims(grayFrame,axis=-1)
    #     print('in loop',ret)

    #     if ret: 
    #         # if video is still left continue creating images 
    #         name = './data/frame'+str(index)+'_' + str(counter) + '.jpg'
    #         #print ('Creating...' + name) 
    #         print('frame',type( grayframe), grayframe.shape)
    #         # writing the extracted images 
    #         numpy_video = np.append(numpy_video,grayframe,axis=-1)
    #         print(numpy_video.shape,type(numpy_video))
    #         #cv2.imwrite(name,  grayFrame) 

    #         # increasing counter so that it will 
    #         # show how many frames are created 
    #         counter += 1
    #         if  counter>30:
    #             break
    #     else: 
    #         break

# Release all space and windows once done 

    # # Calculate RMS
    # rms_window = 1.0 # in seconds 
    # rms = librosa.feature.rms(y=y, hop_length=int(sr*rms_window))
    # rms_db = librosa.core.amplitude_to_db(rms, ref=0.0)
    # print(list(rms_db[0]),rms_db.shape,type(rms_db))

