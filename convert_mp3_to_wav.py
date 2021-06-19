# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:24:24 2021

@author: Adi
"""
import numpy as np
import os
import librosa
import re
import subprocess
import pandas as pd
from tinytag import TinyTag
import sound_similarity

def fix_string(st_in):
    # only lower case
    st0 = st_in.lower() #casefold() ?
    st1 = re.sub("[^0-9a-zA-Z,' ']", "",st0)
    no_good_string =[' hd','hd ','official music video','official','video']
    for bad in no_good_string:
        st1 =st1.replace(bad,'')
    st2 =st1
    st2 = st2.replace(",", "")
    st2 = st2.replace(".", "")
    st2 = st2.replace("'", "")
    # remove all space long then 1
    while '  ' in st2:
        st2 = st2.replace("  ", " ")
    
    if len(st2)<2:  #not english
        st22=st0
        st22 = st22.replace("-", "")
        st22 = st22.replace(":", "")
        st22 = st22.replace("@", "")
        st22 = st22.replace("#", "")
        st22 = st22.replace(",", "")
        st22 = st22.replace(".", "")
        st22 = st22.replace("'", "")
        
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
                  'artist',
                  'title',
                  'albume',
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
phath_to_index_sound_data = main_phath+r'\data\data_sound_sim\index_sound_data.csv'

# path for folder
phath_for_mp3_sound = main_phath+r'\data\data_sound_sim\mp3_sound2'
phath_for_wav_sound = main_phath+r'\data\data_sound_sim\wav_sound2'
phath_for_np_sound = main_phath+r'\data\data_sound_sim\np_sound2'


# parameter for procesing


mp3_file = os.listdir(phath_for_mp3_sound)

mp3_file  = pd.read_csv(phath_to_index_sound_data)['name'].tolist()

for file_name in mp3_file: # loop on video in folder "raw_video"
    # clean song name from char 'Brit go H;Ome'=> 'brit_go_home'
    print('------')
    print(file_name[:-4])
    #file_name_r = fix_string(file_name[:-4])
    file_name_r = file_name
    print(file_name_r)
    print('-------new song:',file_name_r)
    # load index_data (pandas datafram) try to add new song name
    
    sound_mp3_phath = phath_for_mp3_sound+'\\'+file_name
    print('sound_mp3_phath',sound_mp3_phath)
    
    
    flag_name,data_index = new_sound_data(file_name_r,file_name[:-4],phath_to_index_sound_data)
    if flag_name and data_index.at[file_name_r,'no missing data'] : #if name in data, move to next song
        print('song in data, no misissing data, move to next song')
        os.remove(sound_mp3_phath)
        continue
    
    # all name+path of new file
   
    sound_wav_phath = phath_for_wav_sound+'\\'+file_name_r+'.wav'
    print('sound_wav_phath',sound_wav_phath)
    sound_np_phath = phath_for_np_sound+'\\'+file_name_r+'.npy'
    print('sound_np_phath',sound_np_phath)
    print('--------------------start---------------')
    # test if wav in folder
    if (file_name_r+'.wav') not in os.listdir(phath_for_wav_sound):
        print('no wav file')
        try:
            print(sound_mp3_phath)
            print(sound_wav_phath)
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
        print(sound_wav_phath)
        (sig, rate) = librosa.load(sound_wav_phath, sr=sampling_rate)
        np.save(sound_np_phath,  sig)
        print('convert wav file to numpy')
        
    else:
        print('numpy sound file exists')
        rate = sampling_rate
        sig = np.load(sound_np_phath)
    
    #tag = TinyTag.get(sound_mp3_phath)

    
    #data_index.at[file_name_r,'number of bit'] = len(sig)
    #data_index.at[file_name_r,'len[sec]'] = tag.duration
    #data_index.at[file_name_r,'bit for sec'] = len(sig)/data_index.at[file_name_r,'len[sec]']
    #try :
        #data_index.at[file_name_r,'artist'] = fix_string(tag.artist)
        #data_index.at[file_name_r,'title'] = fix_string(tag.title)
        #data_index.at[file_name_r,'albume'] = fix_string(tag.album)
        
    #except:
        #pass
    
    #data_index.at[file_name_r,'path_sound_np'] = sound_np_phath
    #data_index.at[file_name_r,'path_wav'] = sound_wav_phath
    print('1')
    grid = 180
    if data_index.at[file_name_r,'len[sec]'] > 180:
        print('2')
        len_to_index = int(data_index.at[file_name_r,'number of bit']*(180/data_index.at[file_name_r,'len[sec]']))
        sig1 = sig[:len_to_index]
        
    else:
        print('1')
        sig1 = np.zeros(180)
        sig1[:data_index.at[file_name_r,'number of bit']] = sig
    
    print('2')
    p1 = int( (0.3*data_index.at[file_name_r,'number of bit'])//grid)
    print('3')
    v1 = sound_similarity.crop_sig_mean(sig1,p=p1,grid=grid,padding=0)
    print('4')
    #print(v2)
    index1 = sound_similarity.sig_dist_full(v1)
    print('5')
    data_index.at[file_name_r,'index3min'] = v1
    print('1')
    data_index.at[file_name_r,'no missing data'] = True
    print('1')
    data_index.to_csv(phath_to_index_sound_data)
    print('1')
    os.remove(sound_mp3_phath)


    