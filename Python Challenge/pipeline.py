# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:43:47 2020

@author: paschoeto
"""
from moviepy.editor import VideoFileClip
from camera_calibrating import Camera
from color_gradient import ColorGradient
from undistorted_transform import Perspective
from lane_fit import Lane
import numpy as np

import glob
import matplotlib.image as mpimg
import cv2

class Pipeline:
    
    def __init__(self, toCalibrate = False ,src = np.float32([(580, 460), (205, 720), (1110, 720), (703, 460)]), dest = np.float32([(320, 0), (320, 720), (960, 720), (960, 0)])):
        
        self.camera = Camera()
        self.color = ColorGradient(gamma=0.5)
        self.perspective = Perspective(src, dest)
        self.lane = Lane()
        self.toCalibrate = toCalibrate

    def pipeline(self, image):    
    
        # Compute the camera calibration matrix and distortion coefficients given a set of 
        # chessboard images.
        self.camera.computeMatrixCoefs(self.toCalibrate)
            
        #  Apply a distortion correction to raw images.
        self.camera.loadCalibration()
        img, dst = self.camera.applyCalibration(image)
        #self.camera.printUndistortion(img, dst) 
    
        # # Use color transforms, gradients, etc., to create a thresholded binary image.
        combined_binary = self.color.applyColorGradient(dst)
        #self.color.plotColorGradient(dst, combined_binary)
        
        # # Apply a perspective transform to rectify binary image ("birds-eye view").    
        warped, M = self.perspective.applyPerspective(combined_binary, self.camera.getMtx(), self.camera.getDist())
        self.perspective.plotUndist(warped, combined_binary)
        
        # # Apply Lane Calculations
        left_fitx, right_fitx, left_curverad, right_curverad, offset = self.lane.applyLaneCalculations(warped)
        
        # # Warp the detected lane boundaries back onto the original image
        result = self.perspective.applyInversePerpectives(img, dst, warped, np.linalg.inv(M), left_fitx, right_fitx, self.lane.getPloty())
        #self.perspective.plotUndist(result,warped)
        
        #return warped, None, None, None
        return result, left_curverad, right_curverad, offset
    
    def imageProcessing(self, path = "../test_images/*.jpg", outputPath = "../output_images/output"):
        images = glob.glob(path)
        for idx, fname in enumerate(images):
            img = mpimg.imread(fname)
            result, left_curverad, right_curverad, offset = self.pipeline(img)
            result = self.insertFramesValues(result, left_curverad, right_curverad, offset)
            cv2.imwrite(outputPath + str(idx) + ".jpg", result)
            
    def process_image(self,image):
        result, left_curverad, right_curverad, offset = self.pipeline(image)
        self.insertFramesValues(result, left_curverad, right_curverad, offset)
        
        return result
    
    def videoClip(self,  sourceVideo, outputVideo):
        white_output = 'test_videos_output/' + outputVideo
        clip1 = VideoFileClip(sourceVideo)
        white_clip = clip1.fl_image(self.process_image)
        white_clip.write_videofile(white_output, audio=False)
        
    def insertFramesValues(self, img, left_curverad, right_curverad, offset):
        position = (10,50)
        cv2.putText(img,"Offset: " + str(np.round(offset,2)),position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)
        position = (10,90)
        cv2.putText(img,"Left Curve Radius(km): " + str(np.round(left_curverad/1000,1)),position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)
        position = (10,130)
        cv2.putText(img,"Right Curve Radius(km): " + str(np.round(right_curverad/1000,1)),position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)
        
        return img