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



phath_for_url = r'\data\url_youtube.txt'
phath_for_saving_video = r'\data\video'

#url_test = 'https://www.youtube.com/watch?v=4vvBAONkYwI&ab_channel=BritneySpearsBritneySpears'
url_test = open(main_phath+phath_for_url,'r')

print(main_phath+phath_for_url)

for i in url_test: 
    print('i:',i)
    try: 
          
        # object creation using YouTube
        # which was imported in the beginning 
        yt = YouTube(i)
        print(yt.title)
    except: 
          
        #to handle exception
        print("Connection\\url Error")  
        continue  #try next link
      
     
    #set stream resolution
    my_video = yt.streams.get_highest_resolution()

        
    #Download video
    my_video.download(main_phath+phath_for_saving_video)
    
    # get the video with the extension and
    # resolution passed in the get() function 
    
    
print('Task Completed!') 











