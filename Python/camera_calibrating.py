# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:54:24 2020

@author: paschoeto
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

def pipeline_images_undist(img, objpoints, imgpoints):
    
    dist_pickle = {}
    ret, mtx, dist, rvecs, tvecs = calibrate(img, objpoints, imgpoints)
    dst = undistortion(img, mtx, dist)  
    save_calibration(dist_pickle, objpoints, imgpoints, mtx, dist)
    
    return img, dst
        
def calibration_matrix_coefs(path):    
    nx = 9
    ny = 6    
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    
    # Prepare points, like (0,0,0)
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    
    # Make a list of calibration images
    images = glob.glob(path)
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
    
        # convert to Gray Scale
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessborder corner
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        # If corners are found, add objects points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        
    cv2.destroyAllWindows()
    
    return objpoints, imgpoints

def calibrate(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    return ret, mtx, dist, rvecs, tvecs
    
def save_calibration(dist_pickle, objpoints, imgpoints, mtx, dist):    
    # Save the camera calibration result for later use
    dist_pickle["objpoints"] = objpoints
    dist_pickle["imgpoints"] = imgpoints
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "../output_images/wide_dist_pickle.p", "wb"))
    
def undistortion(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
   
    return dst

def load_calibration():
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load( open( "../output_images/wide_dist_pickle.p", "rb" ) )
    
    return dist_pickle
    
def print_cheessboards(img, corners,ret, nx, ny):
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
    plt.imshow(img)
    plt.show()
    
def print_undistortion(img, dst):
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
