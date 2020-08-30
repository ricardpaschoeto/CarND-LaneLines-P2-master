# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:43:47 2020

@author: paschoeto
"""
import camera_calibrating as cc
import color_gradient as cg
import undistorted_transform as ut
import region as r
import lane_fit as lf
import inverse_bird_eye as ibe

import numpy as np
import matplotlib.pyplot as plt


def pipeline(image, calibration=False):
    
    # Compute the camera calibration matrix and distortion coefficients given a set of 
    # chessboard images.
    if(calibration):
        objpoints, imgpoints = cc.calibration_matrix_coefs("../camera_cal/*.jpg")
        
    #  Apply a distortion correction to raw images.
    dist_pickle = cc.load_calibration()
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]
    
    img, dst = cc.pipeline_images_undist(image, objpoints, imgpoints)
    
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    combined_binary = cg.pipeline(dst, s_thresh=(170, 255), sx_thresh=(20, 100)) 
    
    # Region of Interest
    imshape = img.shape
    vertices = np.array([[(150,imshape[0]),(530, 420), (imshape[1]-500, 420), (imshape[1]-50,imshape[0])]], dtype=np.int32)
    masked_edges = r.region_of_interest(combined_binary, vertices)
    #plt.imshow(masked_edges)
    #plt.show()
     
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    warped, M = ut.perspective_pipeline(masked_edges, mtx, dist)
    #ut.plot_undist(img, warped)
    
    # Detect lane pixels and fit to find the lane boundary.
    left_fit, left_fitx, right_fit, right_fitx, ploty, out_img = lf.fit_polynomial(warped)

    # Determine the curvature of the lane and vehicle position with respect to center.
    left_curverad, right_curvera = lf.measure_curvature_real( ploty, left_fit, right_fit)
    #print((left_curverad,right_curvera))
    
    # Warp the detected lane boundaries back onto the original image
    result = ibe.inverse_perpectives(img, dst, warped, np.linalg.inv(M), left_fitx, right_fitx, ploty)
    plt.imshow(result)
    plt.show()
    
    return result



