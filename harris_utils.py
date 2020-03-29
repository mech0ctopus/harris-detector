# -*- coding: utf-8 -*-
"""
Harris Detector Utilities
"""
import numpy as np
import cv2
from scipy import signal, ndimage

def compute_G(fx,fy):
    '''Computes structure matrix G.'''
    G11 = ndimage.gaussian_filter(fx**2, sigma=1)
    G12_21 = ndimage.gaussian_filter(fx*fy, sigma=1)
    G22 = ndimage.gaussian_filter(fy**2, sigma=1)

    G=np.array([[G11, G12_21],
                [G12_21, G22]])
    return G

def compute_response(G,threshold):
    '''Computes Harris response.'''
    # Find determinant
    detA = G[0][0] * G[1][1] - G[1][0] ** 2
    # Find trace
    traceA = G[0][0] + G[1][1]
    harris_response = detA - threshold * traceA ** 2
    return harris_response

def corner_assignment(img_rgb,harris_response):
    '''Returns corners based on r-value.'''
    img = np.copy(img_rgb)
    corner_locations=[]
    for rowindex, response in enumerate(harris_response):
        for colindex, r in enumerate(response):
            if r > 0:
                # This is a corner
                img[rowindex, colindex] = [255,0,0]
                corner_locations.append((rowindex, colindex))
    return img, corner_locations

def get_derivative(f,height,width,direction='x',channels=[0,1,2],show=False):
    '''Estimate derivative fx or fy.'''
    #Assume f==img
    f_prime=np.zeros((height,width))
    #Get convolutional kernal
    kernel=get_kernel(direction)
    #Calculate filtered result
    f_prime=convolution2d(f,kernel).astype(float)
    if show==True:
        cv2.imshow(f'direction: {direction}, Filter',f_prime)
    return f_prime

def load_image(image_path,gray=True,show=False):
    '''Loads image into opencv. Returns img, height, and width.'''
    if gray==True:    
        #Load into grayscale
        img=cv2.imread(image_path,0)
        height,width=img.shape
    else:
        #Load RGB
        img=cv2.imread(image_path)
        height,width=img.shape[:-1]
    if show==True:
        cv2.imshow('image',img)
    return img, width, height

def convolution2d(image, kernel):
    '''Performs 2D convolution.'''
    filtered_image=signal.convolve2d(image, kernel, mode='same')
    return filtered_image

def get_kernel(direction):
    '''Returns corresponding derivative kernel.'''
    #Define convolving derivative kernels
    Dx=np.array([[-1, 0, 1],
                 [-1, 0, 1],
                 [-1, 0, 1]])
    Dy=np.array([[-1, -1, -1],
                 [0, 0, 0],
                 [1, 1, 1]])
    if direction=='x':
        kernel=Dx
    elif direction=='y':
        kernel=Dy 
    else:
        kernel=None
        
    return kernel