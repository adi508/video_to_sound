# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:40:32 2021

@author: Adi
"""
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)



class My_sound():
    
    def __init__(self,sig,per_sec):
        self.sig = sig
        self.len = len(sig)
        self.time = self.len/per_sec

    
    def dis_from(self,sig1,per_sec1,grid):
        
        padd = self.len//(grid+1)
        p = padd//4
        v1 = crop_sig_mean(self.sig,p=p,grid=grid,padding=padd)
        v2 = crop_sig_mean(sig1,p=p,grid=grid,padding=padd)
        dis = np.linalg.norm(v1,v2)
        return dis    
        
    def index_v1(self):
        self.v1grid = 64
        self.padding1 = self.len//(self.v1grid+1)
        self.p1 = self.padding1// 3
        self.sig1 = crop_sig_mean(self.sig1,p=self.p1,grid=self.v1grid,padding=self.padding1)
        self.v1 = sig_dist_full(self.sig1)
    
    def index_v2(self):
        self.v2grid = 256
        self.padding2 = self.len//(self.v2grid+1)
        self.p2 = self.padding2// 3
        self.sig2 = crop_sig_mean(self.sig2,p=self.p2,grid=self.v2grid,padding=self.padding2)
        self.v2 = sig_dist_full(self.sig2)
           

def similarity_sig(sig1,per_sec1,sig2,per_sec2,grid):
    l1 = len(sig1)
    padd1 = l1//(grid+1)
    p1 = padd1//3
    #print(l1,padd1,p1,grid)
    v1 = crop_sig_mean(sig1,p=p1,grid=grid,padding=padd1)
    #print(v1)
    m1 = sig_dist_full(v1)
    #print(m1)
    l2 = len(sig2)
    padd2 = l2//(grid+1)
    p2 = padd2//3
    #print(l2,padd2,p2,grid)
    v2 = crop_sig_mean(sig2,p=p2,grid=grid,padding=padd2)
    #print(v2)
    m2 = sig_dist_full(v2)
    #print(m2) 
    return np.linalg.norm(m1-m2) 
    
def ker_mean_1d(sig,x0=0,p=2):
    index_x = np.arange(p)+x0
    out = sum(sig[index_x])
    return out/p

v_ker_mean_1d = np.vectorize(ker_mean_1d,excluded=['sig'])

def crop_sig_mean(sig,p=3,grid=9,padding=1):
    l = len(sig)
    #print(l,p,grid,padding)
    while (True):
        step = (l-2*padding-p+1)//(grid-1)
        #print('step:',step)
        #print('step_w:',step_w,'step_h:',step_h)
        index = np.arange(padding,l-padding,step)
        l0= index.shape[0]
        if  l0==grid:
            break
        padding+=1
    
    #print('real padding',padding)
    out1 = v_ker_mean_1d(sig = sig,x0=index,p=p)
    #out1 = out1.reshape((grid,grid))
    return out1

def sig_dist_small(all_sig,x0=1): 
    sig5 = all_sig[(x0-2):(x0+3)]
    trash = np.std(sig5)*0.25

    out = sig5-sig5[2]
    out_p = out>trash
    out_n = out<-trash
    out_r = 1*out_p-1*out_n
    #print(np.delete(out_r,3, None)+1)
    return np.delete(out_r,3, None)+1

v_sig_dist_small = np.vectorize(sig_dist_small,excluded=['all_sig'],otypes=[np.ndarray])

    
def sig_dist_full(sig5):  
    l = len(sig5)
    sig_pad = np.zeros((l+4))
    sig_pad[2:-2] = sig5 
    index = np.arange(2,l+2)           
    out = v_sig_dist_small(all_sig = sig_pad,x0=index)
    out = np.hstack(out[:]).astype(np.int8)
    
    return(out)


def main():
    
    sig1 = np.random.randint(0,255,size =(128*24))
    sig2 = np.random.randint(0,255,size =(128*24))
    print('hiiiiiiiiiiiiii????')
    print(similarity_sig(sig1,24,sig2,24,15))
    

#main()
print('upload sound_similarity...')
print('    main function : similarity_sig(sig1,per_sec1,sig2,per_sec2,grid)')
print('    main class: My_sound(sig,per_sec) ')














