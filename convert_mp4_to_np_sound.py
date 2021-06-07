# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:28:23 2021

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

def pyramid_pass_sum(vx_in,vy_in,start,end):
    # find index of start point and end point
    index_start = np.searchsorted(vx_in,start)
    index_end = np.searchsorted(vx_in,end)
    # finde index of mid point
    index_mid = int(0.5*(index_start+index_end))
    # restart mask for filter, all zero
    mask = np.zeros_like(vy_in)
    mask[index_mid]= 1 #mid filter is 1
    
    # left side of filter linar function, start in 0 end in 1
    step =2/(index_end-index_start)   
    for index in np.arange(index_start,index_mid):
        mask[index]= mask[index-1]+step
    # rifht side of filter linar function, start in 1 end in 0   
    for index in np.arange(index_mid,index_end):
        mask[index]= mask[index-1]-step
    
    # inner product of mask with value
    scalar = np.inner(mask,vy_in)
    return scalar

def mel_filter(vx_in,vy_in,number_filter = 26):
    # define mel filter function and oposite function
    mel_f = lambda x:1125*np.log(1+x/700)
    mel_o_f = lambda x:700*(np.exp(x/1125)-1)
    # transpose array by mel 
    vx_log = mel_f(vx_in)
    # divaide space by the new log base
    points_log = np.linspace(0,vx_log[-1],number_filter+1)
    points = end_poits = mel_o_f(points_log)
    
    # define start point of evry filter
    start_poits = points[:-1]
    # define end point of evry filter
    end_poits = points[1:]
    
    list_out = []

    for start,end in zip(start_poits, end_poits):
        list_out.append( pyramid_pass_sum(vx_in,vy_in,start,end)  )
    
    
    v_out = np.array(list_out)
    return v_out

def cut_sig_to_win(sig,win_size,step):
    # cut signals to windowes
    # f([1,2,3,4],2,1) = [[1,2],[2,3],[3,4]]
    # f([1,2,3,4,5,6],3,2) = [[1,2,3],[3,4,5],[5,6,0]]
    new_len =1+ (len(sig)-win_size)/step
    while new_len>(new_len//1):
        sig = np.append(sig,[0]) #pad sig
        new_len =1+ (len(sig)-win_size)/step
        
    out=[]
    
    for index in range(0,len(sig)-win_size+step,step):
        out.append(sig[index:(index+win_size)])
    out = np.array(out)
    
    return out.T

def fft_np(sig,sr):
    n = len(sig)
    freq = np.fft.rfftfreq(n,d=1/rate)
    out = np.abs(np.fft.rfft(y)/n)
    return out,freq    

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
    
    
    
    # Calculate RMS
    print(sound_wav_phath)
    # parameters for converting wav to vector
    sampling_rate = 32768  # [Hz]  2^15
    
    
    (sig, rate) = librosa.load(sound_wav_phath, sr=sampling_rate)
    len_in_sec = librosa.get_duration(y=sig, sr=rate)
    print('----len_in_sec:',len_in_sec)
    
    #   clean signal
    sig = (sig*32767).astype(int)
    sig = sig/32767.0
    
    print('---sig:')
    print('max',max(sig),'min:',min(sig),'shape:',sig.shape,type(sig))
    
    print('---rata:',rate,type(rate),'sampel in minute')
    
    # Parameters for STFT - Short-Time Fourier Transform
    window_length = 0.025 #[sec] ==> number of sampel in window: sampling_rate*window_length
    window_step = 0.01    #[sec] ==> number of sampel in step: sampling_rate*window_step
    NFFT = 1025 # number of samples in window f(n)=1+2^n
    
    
        

    
   
    
    
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