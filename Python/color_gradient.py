# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:34:05 2020

@author: paschoeto
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

def pipeline(img, s_thresh, sx_thresh):
    
    #combined_binary = color_gradient(img, s_thresh, sx_thresh)
    gray = grayscale(img)    
    darked = darkedgray(gray, gamma=0.3)
    combined_binary = hlsscale(img, darked)
    
    return combined_binary
        
# Edit this function to create your own pipeline.
def color_gradient(img, s_thresh, sx_thresh):
    img = np.copy(img)
    width = img.shape[1]
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    sxbinary[:, :width//2] = 0	# use the left side
    s_binary[:,width//2:] = 0 # use the right side
    
    combined_binary = sxbinary | s_binary
    # Combine the two binary thresholds
    # combined_binary = np.zeros_like(sxbinary)
    # combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary

def hlsscale(img, darkg):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    lower_white = np.array([0,200,0], dtype="uint8")
    upper_white = np.array([200,255,255], dtype="uint8")
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([10,0,100], dtype="uint8")
    upper_yellow = np.array([40, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    colored_img = cv2.bitwise_and(darkg, darkg, mask=mask)
    
    return colored_img

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def darkedgray(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
    
def plot_images(image, result):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)
    
    ax2.imshow(result, cmap='gray')
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
