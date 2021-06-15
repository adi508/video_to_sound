# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:40:32 2021

@author: Adi
"""
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)



class my_image():
    
    def __init__(self,img):
        h,w = img.shape
        self.img = img
        self.h = h
        self.w = w
        self.index_v1()
        self.index_v2()
    
    def index_v1(self):
        self.v1grid = 4
        self.padding1 = np.min([self.h,self.h])//(self.v1grid+1)
        self.p1 = self.padding1// 2
        self.img1 = crop_img_mean(self.img,p=self.p1,grid=self.v1grid,padding=self.padding1)
        self.v1 = img_dist_full(self.img1)
        #print('v1:',self.v1)
        
    
    def index_v2(self):
        self.v2grid = 9
        self.padding2 = np.min([self.h,self.h])//(self.v2grid+1)
        self.p2 = self.padding2// 2
        self.img2 = crop_img_mean(self.img,p=self.p2,grid=self.v2grid,padding=self.padding2)
        self.v2 = img_dist_full(self.img2)
       # print('v2:',self.v2)
    

def similarity(img1,img2): 
    dis1= img1.v1-img2.v1
    dis2= img1.v2-img2.v2
    out1 = np.linalg.norm(dis1)
    out2 = np.linalg.norm(dis2)
    return out1,out2
    
def ker_mean(img,x0=0,y0=0,p=2):
    index_x = np.arange(p)+x0
    index_y = np.arange(p)+y0
    index_h,index_w = np.meshgrid(index_x,index_y)
    index_w =index_w.flatten()
    index_h =index_h.flatten()
    out = sum(img[index_w,index_h])
    return out/(p*p)

v_ker_mean = np.vectorize(ker_mean,excluded=['img'])

def crop_img_mean(img,p=3,grid=9,padding=1):
    w,h = img.shape
    step_w = (w-padding-p)//(grid)
    step_h = (h-padding-p)//(grid)
    index_w = np.expand_dims(np.arange(padding,w-padding-p,step_w),axis=-1)   
    print(w,h,p,padding)
    index_h = np.expand_dims(np.arange(padding,h-padding-p,step_h),axis=0)
    print(index_w.shape,index_h.shape)
    print(index_w,index_h)
    index_h1,index_w1 = np.meshgrid(index_h,index_w)
    index_w1 =index_w1.flatten()
    index_h1 =index_h1.flatten()
    print(index_w1.shape,index_h1.shape)
    out1 = v_ker_mean(img = img,x0=index_h1,y0=index_w1,p=p)
    out1 = out1.reshape((grid,grid))
    return out1

def img_dist_small(all_img,x0=1,y0=1):
    
    img33 = all_img[(x0-1):(x0+2),(y0-1):(y0+2)]
    trash = np.std(img33)*0.25

    out = img33-img33[1,1]
    out_p = out>trash
    out_n = out<-trash
    out_r = 1*out_p-1*out_n
    #out_flat = np.expand_dims(np.delete(out_r,4, None),axis=-1)
    #print(out_flat.shape)
    #print(type(np.delete(out_r,4, None)))
    return np.delete(out_r,4, None)+1

v_img_dist_small = np.vectorize(img_dist_small,excluded=['all_img'],otypes=[np.ndarray])

    
def img_dist_full(img99):  
    h,w = img99.shape
    img_pad = np.zeros((h+2,w+2))
    img_pad[1:-1,1:-1] = img99 
    #print(img_pad)
    index_w = np.expand_dims(np.arange(1,w+1),axis=-1)   
    index_h = np.expand_dims(np.arange(1,h+1),axis=0)
    index_h1,index_w1 = np.meshgrid(index_h,index_w)
    index_w1 =index_w1.flatten()
    index_h1 =index_h1.flatten()
    out = v_img_dist_small(all_img = img_pad,x0=index_h1,y0=index_w1)
    out = np.hstack(out[:]).astype(np.int8)
    out = out.reshape(h*w,8)
    return(out)


def main():
    img1 = my_image(np.random.randint(0,255,size =(100,100)))
    img2 = my_image(np.random.randint(0,255,size =(100,100)))
    
    
    
    # img_test.index_v1()
    
    #img_test = np.ones((3,3))
    #print(img_dist_full(img_test))
    print(similarity(img1,img2))
    print(similarity(img2,img2))
    

#main()
















