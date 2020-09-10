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

class Camera:

    def __init__(self, nx=9, ny=6):
        
        self.nx = nx
        self.ny = ny
        # 3D points in real world space
        self.objpoints = []
        # 2D points in image plane
        self.imgpoints = []
        self.dist_pickle = {}
        self.mtx = None
        self.dist = None
        
    def computeMatrixCoefs(self, toCompute):
        if(toCompute):
            self.calibrationMatrixCoefs("../camera_cal/*.jpg")

    def applyCalibration(self, img):        
        self.calibrate(img)
        self.saveCalibration()
        dst = self.undistortion(img)
        
        return img, dst
            
    def calibrationMatrixCoefs(self,path):    
        
        # Prepare points, like (0,0,0)
        objp = np.zeros((self.ny*self.nx, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)
        
        # Make a list of calibration images
        images = glob.glob(path)
        
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
        
            # convert to Gray Scale
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chessborder corner
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            
            # If corners are found, add objects points, image points
            if ret == True:
                self.imgpoints.append(corners)
                self.objpoints.append(objp)
    
    def calibrate(self,img):
        img_size = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size,None,None)
    
    def undistortion(self, img):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
       
        return dst
        
    def saveCalibration(self):    
        # Save the camera calibration result for later use
        self.dist_pickle["objpoints"] = self.objpoints
        self.dist_pickle["imgpoints"] = self.imgpoints
        self.dist_pickle["mtx"] = self.mtx
        self.dist_pickle["dist"] = self.dist
        pickle.dump(self.dist_pickle, open( "../output_images/wide_dist_pickle.p", "wb"))
        
    def loadCalibration(self):
        # Read in the saved objpoints and imgpoints
        self.dist_pickle = pickle.load( open( "../output_images/wide_dist_pickle.p", "rb" ) )
        self.objpoints = self.dist_pickle["objpoints"]
        self.imgpoints = self.dist_pickle["imgpoints"]
        self.mtx = self.dist_pickle["mtx"]
        self.dist = self.dist_pickle["dist"]

    def getMtx(self):
        return self.mtx
    def getDist(self):
        return self.dist
        
    def printCheessboards(self, chessboard, corners,ret):
        # Draw and display the corners
        chessboard = cv2.drawChessboardCorners(chessboard, (self.nx,self.ny), corners, ret)
        plt.imshow(chessboard)
        plt.show()
        
    def printUndistortion(self, img, dst):
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)
