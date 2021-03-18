#!/usr/bin/env python3
#
# Suggested use:
#   from Anomaly import highlightAnomalies
#
import cv2 as cv
import numpy as np


def highlightAnomalies(reference,deviation,mask,anomlimit=0.015):
    #deviation[np.isnan(deviation)]=0 #to suppress a warning on the next line
    deviation = np.abs(deviation).clip(0,anomlimit).reshape(reference.shape[:2])/anomlimit
    nanmask = (~mask).reshape(reference.shape[:2])
    
    h1 = np.zeros_like(reference)
    h1[:,:,2] = deviation*255
    h1[nanmask,:] = (255,0,0)
    
    h2 = np.zeros_like(reference)
    h2[:,:,0] = deviation*255
    h2[:,:,1] = deviation*255
    h2[:,:,2] = deviation*255
    h2[nanmask,:] = 0

    ref2 = cv.subtract(reference,h2)
    out = cv.add(ref2,h1)


    return out
