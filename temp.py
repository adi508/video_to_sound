# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:42:37 2021

@author: Adi
"""
import numpy as np

import matplotlib.pyplot as plt

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


# First create some toy data:
x = np.linspace(0, 10, 400)
f = lambda t:(t*t)-5*t
y = f(x)
# start = 2*np.pi 
# end = 4*np.pi
# print(pirmid_integral(x,y,start,end))

# start = 0
# end = 2*np.pi
# print(pirmid_integral(x,y,start,end))
ans = mel_filter(x,y,number_filter = 200)
print(ans[-1])
print(y[-1])
# # Create just a figure and only one subplot
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title('Simple plot')

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# # Create four polar axes and access them through the returned array
# fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
# axs[0, 0].plot(x, y)
# axs[1, 1].scatter(x, y)

# # Share a X axis with each column of subplots
# plt.subplots(2, 2, sharex='col')

# # Share a Y axis with each row of subplots
# plt.subplots(2, 2, sharey='row')

# # Share both X and Y axes with all subplots
# plt.subplots(2, 2, sharex='all', sharey='all')

# # Note that this is the same as
# plt.subplots(2, 2, sharex=True, sharey=True)

# # Create figure number 10 with a single subplot
# # and clears it if it already exists.
# fig, ax = plt.subplots(num=10, clear=True)