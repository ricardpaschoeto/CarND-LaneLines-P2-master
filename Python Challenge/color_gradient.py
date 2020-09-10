# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:34:05 2020

@author: paschoeto
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ColorGradient:

    def __init__(self, gamma, s_thresh=(170, 255), sx_thresh=(20, 100)):
        
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh
        self.gamma = gamma

    def applyColorGradient(self, img):
        
        gray = self.grayscale(img)    
        darked = self.darkedgray(gray)
        combined_binary = self.hlsscale(img, darked)
        
        return combined_binary
            
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
    
    def plotColorGradient(self,image, result):
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=40)
        
        ax2.imshow(result, cmap='gray')
        ax2.set_title('Pipeline Result', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        
