# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:16:19 2020

@author: paschoeto
"""
import numpy as np

class Line():
    
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
    def setDetected(self,detected):
        self.detected = detected
    
    def getDetected(self):
        return self.detected
    
    def setRecentXfitted(self,recent_xfitted):
        self.recent_xfitted.append(recent_xfitted)
        self.setBestX()
        
    def getRecentXfitted(self):
        return self.recent_xfitted
    
    def setBestX(self):
        self.bestx = np.average(self.recent_xfitted)
    
    def getBestX(self):
        return self.bestx
    
    def setCurrentFit(self,current_fit):
        self.current_fit.append(current_fit)
        self.setBestFit()
    
    def getCurrentFit(self):
        return self.current_fit
    
    def setBestFit(self):
        self.best_fit = np.average(self.current_fit)
        
    def getBestFit(self):
        return self.best_fit
    
    def setRadius(self,radius):
        self.radius_of_curvature = radius
    
    def getRadius(self):
        return self.radius_of_curvature
    
    def setLineBasePos(self,pos):
        self.line_base_pos = pos
        
    def getLineBasePos(self):
        return self.line_base_pos
    
    def setDiffs(self,fit):
        self.diffs = self.best_fit - fit
        
    def getDiffs(self):
        return self.diffs
    
    def setAllx(self,x):
        self.allx = x
    
    def getAllx(self):
        return self.allx
    
    def setAlly(self,y):
        self.ally = y
        
    def getAlly(self):
        return self.ally
        
        
        