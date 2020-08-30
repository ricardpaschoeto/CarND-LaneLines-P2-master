# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:10:15 2020

@author: paschoeto
"""
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def perspective_pipeline(img, mtx, dist):

    warped, M = corners_unwarp(img, mtx, dist)
    
    return warped, M

def corners_unwarp(img, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    gray = cv2.undistort(img, mtx, dist, None, mtx)
    warped = None
    M = None
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([(580, 460), (205, 720), (1110, 720), (703, 460)])
    
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([(320, 0), (320, 720), (960, 720), (960, 0)])
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(gray, M, img_size)

    # Return the resulting image and matrix
    return warped, M

def plot_undist(img, warped):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped)
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
