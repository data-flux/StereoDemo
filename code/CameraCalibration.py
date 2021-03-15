#
# Suggested use:
#   from CameraCalibration import undistort,verticalShift
#
import numpy as np
import cv2 as cv

def undistort_l(input_img):
    mtx_l = np.array([[4.59047907e+03, 0.00000000e+00, 9.43169501e+02],
                      [0.00000000e+00, 4.58685720e+03, 6.83044874e+02],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_l = np.array([[-1.68710942e-01,  
                         1.59061957e+00,  
                         4.84148293e-03,
                        -1.76701105e-04,
                        -2.60947845e+01]])
    newcameramtx_l = np.array([[4.52489404e+03, 0.00000000e+00, 9.42041238e+02],
                               [0.00000000e+00, 4.52736230e+03, 6.85554034e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    return cv.undistort(input_img, mtx_l, dist_l, None, newcameramtx_l)


def undistort_r(input_img):
    mtx_r = np.array([[4.57357104e+03, 0.00000000e+00, 9.61683241e+02],
                      [0.00000000e+00, 4.57051256e+03, 6.84033444e+02],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_r = np.array([[-2.30108491e-01,
                         4.25795521e+00, 
                         4.08540567e-03, 
                         3.06190976e-04,
                        -5.87346437e+01]])
    newcameramtx_r = np.array([[4.49798096e+03, 0.00000000e+00, 9.61499484e+02],
                               [0.00000000e+00, 4.50455176e+03, 6.86580755e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    return cv.undistort(input_img, mtx_r, dist_r, None, newcameramtx_r)

def verticalShift(input_img, shift,**kargs):
    affine = np.array(  [[ 1.00,  0.00,  0.00],
                         [ 0.00,  1.00, shift]])
    return cv.warpAffine(input_img,affine,input_img.shape[1::-1],**kargs)

undistort = undistort_r
