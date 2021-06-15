# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:42:37 2021

@author: Adi
"""
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import image_similarity

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
    new_len =1+ (len(sig)-win_size)/step
    while new_len>(new_len//1):
        sig = np.append(sig,[0]) #pad sig
        new_len =1+ (len(sig)-win_size)/step
        
    out=[]
    
    for index in range(0,len(sig)-win_size+step,step):
        out.append(sig[index:(index+win_size)])
    out = np.array(out)
    
    return out

def cut_sig_to_win2(sig,win_size,step):
    new_len =1+ (len(sig)-win_size)/step
    while new_len>(new_len//1):
        sig = np.append(sig,[0]) #pad sig
        new_len =1+ (len(sig)-win_size)/step
    
    new_len = int(new_len)    
    out=np.zeros((win_size,new_len))
    #print(out.T)
    #print(out.T.shape)
    for index in range(win_size):
        temp_index = np.arange(index,index+(new_len)*step,step)
        #print(index,sig[temp_index],temp_index)
        #print(out[index])
        out[index] =  sig[temp_index]
    
    out = out.T
    return out

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

def restart_index_data(path):
    columns_name=['name',
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

def new_data(name,path):
    df1 = pd.read_csv(path_index_data).set_index('name')
    df2 = pd.DataFrame([name],columns=['name']).set_index('name')
    df1 = df1.append(df2)
    df1.to_csv(path)
    

img1 = image_similarity.my_image(np.random.randint(0,255,size =(100,100)))

img2 = image_similarity.my_image(np.random.randint(0,255,size =(100,100)))



# img_test.index_v1()

#img_test = np.ones((3,3))
#print(img_dist_full(img_test))
print(image_similarity.similarity(img1,img2))
print(image_similarity.similarity(img2,img2))