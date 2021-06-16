# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:40:32 2021

@author: Adi
"""
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)



class My_image():
    
    def __init__(self,img):
        h,w = img.shape
        self.img = img
        self.h = h
        self.w = w
        #self.index_v1()
        #self.index_v2()
    
    def dis_from(self,img1,grid):
        
        padd = np.min([self.h,self.w])//(grid+1)
        p = padd//4
        v1 = crop_img_mean(self.img,p=p,grid=grid,padding=padd)
        v2 = crop_img_mean(img1,p=p,grid=grid,padding=padd)
        dis = np.linalg.norm(v1,v2)
        return dis    
        
    def index_v1(self):
        self.v1grid = 3
        self.padding1 = np.min([self.h,self.w])//(self.v1grid+1)
        self.p1 = self.padding1// 3
        self.img1 = crop_img_mean(self.img,p=self.p1,grid=self.v1grid,padding=self.padding1)
        self.v1 = img_dist_full(self.img1)
        #print('v1:',self.v1)
        
    
    def index_v2(self):
        self.v2grid = 9
        self.padding2 = np.min([self.h,self.w])//(self.v2grid+1)
        self.p2 = self.padding2// 3
        self.img2 = crop_img_mean(self.img,p=self.p2,grid=self.v2grid,padding=self.padding2)
        self.v2 = img_dist_full(self.img2)
       # print('v2:',self.v2)
    

def similarity(img1,img2,grid):
    w1,h1 = img1.shape
    padd1 = np.min([h1,w1])//(grid+1)
    p1 = padd1//3
    v1 = crop_img_mean(img1,p=p1,grid=grid,padding=padd1)
    
    w2,h2 = img2.shape
    padd2 = np.min([h2,w2])//(grid+1)
    p2 = padd2//3
    v2 = crop_img_mean(img2,p=p2,grid=grid,padding=padd2)
    
    return np.linalg.norm(v1-v2)
    
    
    
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
    while (True):
        step_w = (w-2*padding-p+1)//(grid-1)
        step_h = (h-2*padding-p+1)//(grid-1)
        #print('step_w:',step_w,'step_h:',step_h)
        index_w = np.arange(padding,w-padding,step_w)
        #print('w:',w,'h:',h,'p:',p,'padding:',padding)
        index_h = np.arange(padding,h-padding,step_h)
        wl= index_w.shape[0]
        wh= index_h.shape[0]
        if wl==grid and wh==grid:
            break
        padding+=1
        

    index_w = np.expand_dims(index_w,axis=-1)   
    index_h = np.expand_dims(index_h,axis=0)    
    index_h1,index_w1 = np.meshgrid(index_h,index_w)
    index_w1 = index_w1.flatten()
    index_h1 = index_h1.flatten()
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
    return np.delete(out_r,4, None)+1

v_img_dist_small = np.vectorize(img_dist_small,excluded=['all_img'],otypes=[np.ndarray])

    
def img_dist_full(img99):  
    h,w = img99.shape
    img_pad = np.zeros((h+2,w+2))
    img_pad[1:-1,1:-1] = img99 
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
    img1 = np.random.randint(0,255,size =(1,128,72))
    img2 = np.random.randint(0,255,size =(1,384,216))
    print('hiiiiiiiiiiiiii????')
    #cv2.imshow('img',img1)
    
    while(False):
        img1 = np.random.randint(0,255,size =(128,72)).astype(np.uint8)
        cv2.imshow('hi', img1)
        
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print (k) # else print its value
    cv2.destroyAllWindows()
    #print(similarity(img1,img2,15))
    

#main()
















