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
# import ffmpeg

def fix_string(st_in):
    # only lower case
    st0 = st_in.lower() #casefold() ?
    st1 = re.sub("[^0-9a-zA-Z,' ']", "",st0)
    no_good_string =[' hd','hd ','official music video','official','video']
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
                  'len[sec]',
                  'number of bit',
                  'bit for sec',
                  'number of farme',
                  'frame for sec',
                  'url',
                  'path_mp4',
                  'path_wav',
                  'path_sound_np',
                  'path_video_np',
                  'error']
    
    index_data = pd.DataFrame(columns=columns_name).set_index('name')
    index_data.to_csv(path)

def new_data(name,old_name,path):
    df1 = pd.read_csv(path).set_index('name')
    if name in df1.index:
        print('erorr!! song name is in data frame')
        print('!!!!!!!!!!!!!!!!')
        return True,np.nan
    df2 = pd.DataFrame([[name,old_name,True]],columns=['name','old_name','error']).set_index('name')
    df1 = df1.append(df2)
    df1.to_csv(path)
    return False,df1

main_phath =r'D:\github\video_to_sound\video_to_sound'

# path for file
phath_for_url = main_phath+r'\data\url_youtube.txt'
phath_to_index_data = main_phath+r'\data\index_all_data.csv'

# path for folder
phath_for_mp4_video = main_phath+r'\data\raw_video'
phath_for_mp3_sound = main_phath+r'\data\mp3_sound'
phath_for_wav_sound = main_phath+r'\data\wav_sound'
phath_for_np_sound = main_phath+r'\data\np_sound'
phath_for_np_video = main_phath+r'\data\np_video'

# parameter for procesing
dim =(128,72)  # dim of frame


restart_index_data(phath_to_index_data)
video_raw_list_name = os.listdir(phath_for_mp4_video)
for file_name in video_raw_list_name:
    # clean song name from char 'Brit go H;Ome'=> 'brit_go_home'
    file_name_r = fix_string(file_name[:-4])
    print('-------new song:',file_name_r)
    # load index_data (pandas datafram) try to add new song name
    flag_name,data_index = new_data(file_name_r,file_name[:-4],phath_to_index_data)
    if flag_name: #if name in data, move to next song
        print('song in data, move to next song')
        continue
    
    # all name+path of new file
    file_in_phath = phath_for_mp4_video+'\\'+file_name
    sound_mp3_phath = phath_for_mp3_sound+'\\'+file_name_r+'.mp3'
    sound_wav_phath = phath_for_wav_sound+'\\'+file_name_r+'.wav'
    sound_np_phath = phath_for_np_sound+'\\'+file_name_r+'.npy'
    video_np_phath = phath_for_np_video+'\\'+file_name_r+'.npy'
    
    # test if mp3 in folder
    if (file_name_r+'.mp3') not in os.listdir(phath_for_mp3_sound):
        print('no mp3 file')
        try:
            video_clip = VideoFileClip(file_in_phath)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(sound_mp3_phath)
        except:
            print('error rading cap to mp3')
            continue
        
    # test if wav in folder
    if (file_name_r+'.wav') not in os.listdir(phath_for_wav_sound):
        print('no wav file')
        try:
            temp_proc = subprocess.call(['ffmpeg', '-i', sound_mp3_phath,sound_wav_phath],shell=True)
        except:
            print('error converting mp3 to wav')
            continue
       
        
    # parameters for converting wav to vector
    sampling_rate = 32768  # [Hz]  2^15
    
    
    # test if np sound in folder
    if (file_name_r+'.npy') not in os.listdir(phath_for_np_sound):
        print('no NP sound file in folder')
        (sig, rate) = librosa.load(sound_wav_phath, sr=sampling_rate)
        np.save(sound_np_phath,  sig)
        
    else:
        print('NP sound file exsist')
        rate = sampling_rate
        sig = np.load(sound_np_phath)
    
    data_index.at[file_name_r,'path_wav'] = sound_wav_phath
    data_index.at[file_name_r,'path_mp4'] = file_in_phath   
    data_index.at[file_name_r,'number of bit'] = len(sig)
    data_index.at[file_name_r,'len[sec]'] = librosa.get_duration(y=sig, sr=rate)
    data_index.at[file_name_r,'bit for sec'] = len(sig)/data_index.at[file_name_r,'len[sec]']
    data_index.at[file_name_r,'path_sound_np'] = sound_np_phath

    data_index.to_csv(phath_to_index_data)
    
    # plot signal [amp/bit]
    max_sig = np.max(sig)
    min_sig = np.min(sig)
    sig = (2*sig-min_sig-max_sig)/(max_sig-min_sig)
    sig = (sig*32767).astype(int)
    sig = sig/32767.0
    sig_s = sig [0:int(0.0051*len(sig))]
    data = np.array([np.arange(len(sig_s)),sig_s]).T
    df_temp = pd.DataFrame(data,columns=['bit', 'amp'])
    ax1 = df_temp.plot.scatter(x='bit', y='amp', c='DarkBlue')
    
    # test if np video in folder
    if (file_name_r+'.npy') not in os.listdir(phath_for_np_video):
        print('no NP video file in folder') 
        # convert mp4 to 3D numpy array [w,h,t] in black and white
        cap = cv2.VideoCapture(file_in_phath)
        counter = 1  #number of farme convert to 3D numpy array
        ret,frame = cap.read()
        
        if not ret:
            print('error in file- cant open with opencv')
            print('erorr reading cap- move to next url')
            # continue to next file
            continue
        
        # restart 3d numpy array
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayFrame, dim, interpolation = cv2.INTER_AREA)
        numpy_video = np.expand_dims(resized,axis=0) # axis=0 for number of frame
        
        # loop over cap, append frame to 3D array
        while(True):
            ret,frame = cap.read() 
            if ret :   #and  counter<100 :
                counter += 1
                # covert frame to gray scale and down-size  it
                grayframe0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grayframe1 = cv2.resize(grayframe0, dim, interpolation = cv2.INTER_AREA)
                # add dim in the first axis so it can be append to all frames in one 3d array
                grayframe = np.expand_dims(grayframe1,axis=0)
                # append frame to 3D array
                numpy_video = np.append(numpy_video,grayframe,axis=0)
                
                # showe frame in win
                cv2.imshow('frame', grayframe1)
                if cv2.waitKey(1) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            
            else:  # end of video/cap
                file_name = phath_for_np_video +'\\'+file_name_r+'.npy'
                np.save(file_name,  numpy_video)
                cap.release()
                cv2.destroyAllWindows()
                break

        np.save(video_np_phath,  numpy_video)    
    else:
        print('np sound file exsist')
        numpy_video = np.load(video_np_phath)
    
    data_index.at[file_name_r,'path_video_np'] = video_np_phath
    data_index.at[file_name_r,'number of farme'] = numpy_video.shape[0]
    data_index.at[file_name_r,'frame for sec'] = np.ceil(
        data_index.at[file_name_r,'number of farme']/data_index.at[file_name_r,'len[sec]']).astype(int)
    data_index.at[file_name_r,'error'] = False
    data_index.to_csv(phath_to_index_data)   
    

    
    
   
    

# Release all space and windows once done 

    # # Calculate RMS
    # rms_window = 1.0 # in seconds 
    # rms = librosa.feature.rms(y=y, hop_length=int(sr*rms_window))
    # rms_db = librosa.core.amplitude_to_db(rms, ref=0.0)
    # print(list(rms_db[0]),rms_db.shape,type(rms_db))

