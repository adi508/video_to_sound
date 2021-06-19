# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:24:24 2021

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

def fft_np(sig,sr):
    n = len(sig)
    freq = np.fft.rfftfreq(n,d=1/rate)
    out = np.abs(np.fft.rfft(y)/n)
    return out,freq    

def restart_index_sound_data(path):
    columns_name=['name',
                  'old_name',
                  'len[sec]',
                  'number of bit',
                  'bit for sec',
                  'frame for sec',
                  'path_wav',
                  'path_sound_np',
                  'no missing data']
    
    index_data = pd.DataFrame(columns=columns_name).set_index('name')
    index_data.to_csv(path)

def new_sound_data(name,old_name,path):
    df1 = pd.read_csv(path).set_index('name')
    if name in df1.index:
        print('song name is in data frame')
        return True ,df1
    df2 = pd.DataFrame([[name,old_name,False]],columns=['name','old_name','no missing data']).set_index('name')
    df1 = df1.append(df2)
    df1.to_csv(path)
    return False,df1

main_phath =r'D:\github\video_to_sound\video_to_sound'

# path for file
phath_to_index_data = main_phath+r'\data\index_sound_data.csv'

# path for folder
phath_for_mp3_sound = main_phath+r'\data\mp3_sound2'
phath_for_wav_sound = main_phath+r'\data\wav_sound2'
phath_for_np_sound = main_phath+r'\data\np_sound2'


# parameter for procesing



restart_index_video_data(phath_to_index_data)
video_raw_list_name = os.listdir(phath_for_mp3_sound)

for file_name in video_raw_list_name: # loop on video in folder "raw_video"
    # clean song name from char 'Brit go H;Ome'=> 'brit_go_home'
    file_name_r = fix_string(file_name[:-4])
    print('-------new song:',file_name_r)
    # load index_data (pandas datafram) try to add new song name
    flag_name,data_index = new_video_data(file_name_r,file_name[:-4],phath_to_index_data)
    if flag_name and data_index.at[file_name_r,'no missing data'] : #if name in data, move to next song
        print('song in data, no misissing data, move to next song')
        continue
    
    # all name+path of new file
    sound_mp3_phath = phath_for_mp3_sound+'\\'+file_name_r+'.mp3'
    sound_wav_phath = phath_for_wav_sound+'\\'+file_name_r+'.wav'
    sound_np_phath = phath_for_np_sound+'\\'+file_name_r+'.npy'
    video_np_phath = phath_for_np_video+'\\'+file_name_r+'.npy'
  
    # test if wav in folder
    if (file_name_r+'.wav') not in os.listdir(phath_for_wav_sound):
        print('no wav file')
        try:
            temp_proc = subprocess.call(['ffmpeg', '-i', sound_mp3_phath,sound_wav_phath],shell=True)
            print('convert mp3 to wav')
        except:
            print('error converting mp3 to wav')
            continue
           
    # parameters for converting wav to vector
    sampling_rate = 32768  # [Hz]  2^15
    
    
    # test if np sound in folder
    if (file_name_r+'.npy') not in os.listdir(phath_for_np_sound):
        print('no numpy sound file in folder')
        (sig, rate) = librosa.load(sound_wav_phath, sr=sampling_rate)
        np.save(sound_np_phath,  sig)
        print('convert wav file to numpy')
        
    else:
        print('numpy sound file exists')
        rate = sampling_rate
        sig = np.load(sound_np_phath)
    
    data_index.at[file_name_r,'path_wav'] = sound_wav_phath
    data_index.at[file_name_r,'number of bit'] = len(sig)
    data_index.at[file_name_r,'len[sec]'] = librosa.get_duration(y=sig, sr=rate)
    data_index.at[file_name_r,'bit for sec'] = len(sig)/data_index.at[file_name_r,'len[sec]']
    data_index.at[file_name_r,'path_sound_np'] = sound_np_phath

    data_index.to_csv(phath_to_index_data)



    