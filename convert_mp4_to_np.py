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
import re
from moviepy.editor import *
import subprocess
import pandas as pd
#import ffmpeg

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

def fft_np(sig,sr):
    n = len(sig)
    freq = np.fft.rfftfreq(n,d=1/rate)
    out = np.abs(np.fft.rfft(y)/n)
    return out,freq    

def restart_index_data(path):
    columns_name=['name',
                  'old_name',
                  'len',
                  'frane_time', #[sec]
                  'number of farme',
                  'number of bit',
                  'url',
                  'path_mp4',
                  'path_wav',
                  'path_sound_np',
                  'path_video_np']
    
    index_data = pd.DataFrame(columns=columns_name).set_index('name')
    index_data.to_csv(path)

def new_data(name,old_name,path):
    df1 = pd.read_csv(path).set_index('name')
    if name in df1.index:
        print('erorr!! song name is in data frame')
        print('!!!!!!!!!!!!!!!!')
        return True,np.nan
    df2 = pd.DataFrame([[name,old_name]],columns=['name','old_name']).set_index('name')
    df1 = df1.append(df2)
    df1.to_csv(path)
    return False,df1

main_phath =r'D:\github\video_to_sound\video_to_sound'

phath_for_url = main_phath+r'\data\url_youtube.txt'
phath_for_mp4_video = main_phath+r'\data\raw_video'
phath_for_mp3_sound = main_phath+r'\data\mp3_sound'
phath_for_wav_sound = main_phath+r'\data\wav_sound'
phath_for_np_sound = main_phath+r'\data\np_sound'
phath_for_np_video = main_phath+r'\data\np_video'
phath_for_saving_video_as_numpy = main_phath+r'\data\matrix_video'
phath_to_index_data = main_phath+r'\data\index_all_data.csv'


restart_index_data(phath_to_index_data)
video_raw_list_name = os.listdir(phath_for_mp4_video)
for file_name in video_raw_list_name:
    file_name_r = fix_string(file_name[:-4])
    
    flag_name,df = new_data(file_name_r,file_name[:-4],phath_to_index_data)
    if flag_name:
        continue
    # step 1  convert sound to np
    file_in_phath =phath_for_mp4_video+'\\'+file_name
    sound_mp3_phath =  phath_for_mp3_sound+'\\'+file_name_r+'.mp3'
    sound_wav_phath =  phath_for_wav_sound+'\\'+file_name_r+'.wav'
    sound_np_phath =  phath_for_np_sound+'\\'+file_name_r+'.np'
    print('3')
    if (file_name_r+'.mp3') not in os.listdir(phath_for_mp3_sound):
        video_clip = VideoFileClip(file_in_phath)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(sound_mp3_phath)
    print('2')
    if (file_name_r+'.wav') not in os.listdir(phath_for_wav_sound):
        subprocess.call(['ffmpeg', '-i', sound_mp3_phath,sound_wav_phath])
    
    df.at[file_name_r,'path_wav'] = sound_wav_phath
    df.at[file_name_r,'path_mp4'] = file_in_phath
    df.at[file_name_r,'url'] = '???'
    
                 
              
  
    # Calculate RMS
    print(sound_wav_phath)
    # parameters for converting wav to vector
    sampling_rate = 32768  # [Hz]  2^15
    
    
    
    (sig, rate) = librosa.load(sound_wav_phath, sr=sampling_rate)
    
 
    len_in_sec = librosa.get_duration(y=sig, sr=rate)
    
    df.at[file_name_r,'len'] = len_in_sec
    df.to_csv(phath_to_index_data)
    print('----len_in_sec:',len_in_sec)
    max_sig = np.max(sig)
    min_sig = np.min(sig)
    sig = (2*sig-min_sig-max_sig)/(max_sig-min_sig)
    sig = (sig*32767).astype(int)
    sig = sig/32767.0
    
    data = np.array([np.arange(len(sig)),sig]).T
    df_temp = pd.DataFrame(data,columns=['length', 'width'])
    ax1 = df_temp.plot.scatter(x='length', y='width', c='DarkBlue')
    
    print('---sig:')
    print('max',max(sig),'min:',min(sig))
    print(sig.shape,type(sig))
    
    print('---rata:',rate,type(rate))
    
    # Parameters for STFT - Short-Time Fourier Transform
    window_length = 0.025 #[sec] ==> number of sampel in window: sampling_rate*window_length
    window_step = 0.01    #[sec] ==> number of sampel in step: sampling_rate*window_step
    NFFT = 1025 # number of samples in window f(n)=1+2^n
    
    d = librosa.stft(sig)  # STFT of y
    print('---d:')
    print('max',np.max(d),'min:',np.min(d))
    print(type(d),d.shape)
    
    s_db = librosa.amplitude_to_db(np.abs(d), ref=np.max)
    print('---s_db:')
    print('max',np.max(s_db),'min:',np.min(s_db))
    print(type(s_db),s_db.shape)
    counter = 0
    f_0 = 0
    # for f_temp in s_db.T:
        
    #     frequency = np.linspace(0,int(0.5*rate),num = len(f_temp))
    #     fre = np.fft.rfftfreq(len(f_temp),d=1/rate)
    #     y = np.fft.rfft(f_temp)
    #     # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)       
    #     #f_temp = f_temp
    #     # ax1.bar(frequency,f_temp)
    #     # ax2.bar(frequency,f_temp-f_0)
    #     # ax1.set_title(counter)
    #     #print(frequency.shape,f_temp.shape)
        
    #     #fig,ax = plt.subplots(1, 1, sharey=True)
    #     #ax.bar(frequency,f_temp)#-f_0)
    #     ax.set_title(counter)
        
    #     plt.pause(0.01)        
    #     counter += 1
    #     f_0 = f_temp
    #     if  counter>100:
    #         break
        

    
   
    
    
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

