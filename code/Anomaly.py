#!/usr/bin/env python3
#
# Suggested use:
#   from Anomaly import flattenSurface, highlightAnomalies
#
import cv2
import numpy as np
import warnings


def highlightAnomalies(reference,deviation,mask,anomlimit=0.0025):
    deviation[np.isnan(deviation)]=0 #to suppress a warning on the next line
    devmask = (np.abs(deviation)>=anomlimit).reshape(reference.shape[:2])
    nanmask = (~mask).reshape(reference.shape[:2])
    highlighted = np.array(reference)
    highlighted[devmask,:] = (0,0,255)
    highlighted[nanmask,:] = (255,0,0)

    return highlighted
