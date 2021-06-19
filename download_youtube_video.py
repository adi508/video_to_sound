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

main_phath = r'D:\github\video_to_sound\video_to_sound'

    
phath_for_url = main_phath+ r'\data\url_youtube.txt'
phath_for_saving_video = main_phath+ r'\data\raw_video'
phath_for_csv_index_data = main_phath+ r'\data\index_all_data.csv'

url_txt = open(phath_for_url,'r')
url_txt = url_txt.read().split(',')

print(main_phath+phath_for_url)
download_flag = True
for index,singal_url in enumerate(url_txt):

    print('-------------------------',index,'---------new video-------')
    try:   
       # object creation using YouTube
       yt = YouTube(singal_url)
       yt = YouTube(yt.watch_url)
       old_title = yt.title
       print(old_title)
    except:
        print(singal_url)        
        print("Connection\\url Error")  
        continue  #try next link
        
    old_title = old_title.replace("'",'')
    old_title = old_title.replace(".",'')
    old_title = old_title.replace(",",'')
    #print('old_title',old_title)
    #print(os.listdir(phath_for_saving_video))
    
    if (old_title+".mp4") not in os.listdir(phath_for_saving_video):
        print('no mp4 file')
        print(old_title+".mp4")
        
        if download_flag:
            #set stream resolution
            my_video = yt.streams.get_highest_resolution()
            #Download video
            try:
                my_video.download(phath_for_saving_video)
                print('download mp4 file')
            except:
                try:
                    my_video.download(phath_for_saving_video)
                    print('download mp4 file',type(my_video))
                except:
                    print('error!')
                    print(" cant- download mp4 file") 
                    print('singal_url')
                    
    
    

    
    
   
    
    
print('Task Completed!') 











