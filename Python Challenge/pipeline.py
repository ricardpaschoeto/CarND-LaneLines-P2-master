# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:43:47 2020

@author: paschoeto
"""
import numpy as np

def pipeline(image, camera, color, perspective, lane, toCompute):    

    # Compute the camera calibration matrix and distortion coefficients given a set of 
    # chessboard images.
    camera.computeMatrixCoefs(toCompute)
        
    #  Apply a distortion correction to raw images.
    camera.loadCalibration()
    img, dst = camera.applyCalibration(image)
    #camera.printUndistortion(img, dst) 

    # # Use color transforms, gradients, etc., to create a thresholded binary image.
    combined_binary = color.applyColorGradient(dst)
    #color.plotColorGradient(dst, combined_binary)
    
    # # Apply a perspective transform to rectify binary image ("birds-eye view").    
    warped, M = perspective.applyPerspective(combined_binary, camera.getMtx(), camera.getDist())
    #perspective.plotUndist(combined_binary, warped)  
    
    # # Apply Lane Calculations
    left_fitx, right_fitx, left_curverad, right_curverad, offset = lane.applyLaneCalculations(warped)
    
    # # Warp the detected lane boundaries back onto the original image
    result = perspective.applyInversePerpectives(img, dst, warped, np.linalg.inv(M), left_fitx, right_fitx, lane.getPloty())
    #perspective.plotUndist(result, left_curverad, right_curverad, offset)
    
    return result, left_curverad, right_curverad, offset



