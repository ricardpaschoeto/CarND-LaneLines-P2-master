# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:34:05 2020

@author: paschoeto
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ColorGradient:

    def __init__(self, gamma, s_thresh=(90, 255), sx_thresh=(30, 100)):
        
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh
        self.gamma = gamma

    def applyColorGradient(self, img):
        
        abs_bin = self.abs_sobel_thresh(img, orient='x', thresh_min=30, thresh_max=100)
        mag_bin = self.mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
        dir_bin = self.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
        hls_bin = self.hls_thresh(img, thresh=(90, 255))
        
        combined = np.zeros_like(dir_bin)
        combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1 ] = 1
          
        return combined

            
    def grayscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def darkedgray(self, img):
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
     
        # apply gamma correction using the lookup table
        return cv2.LUT(img, table)
    
    def hlsscale(self,img, darkg):
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
    
    def colorGradient(self,img):
        img = np.copy(img)
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
        sxbinary[(scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1
        # Stack each channel
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        
        return color_binary

    def abs_sobel_thresh(self,img, orient='x', thresh_min=20, thresh_max=100):
    	"""
    	Takes an image, gradient orientation, and threshold min/max values
    	"""
    	# Convert to grayscale
    	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    	# Apply x or y gradient with the OpenCV Sobel() function
    	# and take the absolute value
    	if orient == 'x':
    		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    	if orient == 'y':
    		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    	# Rescale back to 8 bit integer
    	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    	# Create a copy and apply the threshold
    	binary_output = np.zeros_like(scaled_sobel)
    	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    	# Return the result
    	return binary_output
    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(30, 100)):
    	"""
    	Return the magnitude of the gradient
    	for a given sobel kernel size and threshold values
    	"""
    	# Convert to grayscale
    	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    	# Take both Sobel x and y gradients
    	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    	# Calculate the gradient magnitude
    	gradmag = np.sqrt(sobelx**2 + sobely**2)
    	# Rescale to 8 bit
    	scale_factor = np.max(gradmag)/255
    	gradmag = (gradmag/scale_factor).astype(np.uint8)
    	# Create a binary image of ones where threshold is met, zeros otherwise
    	binary_output = np.zeros_like(gradmag)
    	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
    	# Return the binary image
    	return binary_output

    def dir_threshold(self,img, sobel_kernel=3, thresh=(0, np.pi/2)):
    	"""
    	Return the direction of the gradient
    	for a given sobel kernel size and threshold values
    	"""
    	# Convert to grayscale
    	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    	# Calculate the x and y gradients
    	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    	# Take the absolute value of the gradient direction,
    	# apply a threshold, and create a binary image result
    	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    	binary_output =  np.zeros_like(absgraddir)
    	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    
    	# Return the binary image
    	return binary_output
    
    def hls_thresh(self,img, thresh=(100, 255)):
    	"""
    	Convert RGB to HLS and threshold to binary image using S channel
    	"""
    	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    	s_channel = hls[:,:,2]
    	binary_output = np.zeros_like(s_channel)
    	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        
    	return binary_output

    def plotColorGradient(self,image, result):
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=40)
        
        ax2.imshow(result, cmap='gray')
        ax2.set_title('Pipeline Result', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        
