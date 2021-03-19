#!/usr/bin/env python3
#
# Suggested use:
#   from Surface import cylindricalCoordinates, ransacSinoidFit, flattenSurface
#
import numpy as np
from funcy import retry

def cylindricalCoordinates(xyzcoord):
    import warnings
    ccoord = np.empty_like(xyzcoord)
    ccoord[2,:] = xyzcoord[2,:]
    ccoord[0,:] = np.sqrt(xyzcoord[0,:]**2 + xyzcoord[1,:]**2)
    ccoord[1,:] = np.arctan2(xyzcoord[1,:],xyzcoord[0,:])
    min_angle = -np.pi
    offset = np.pi/2
    ccoord[1,:] -= offset
    with warnings.catch_warnings():
        # suppress a warning over possible NaN still in ccoord
        warnings.simplefilter("ignore")
        ccoord[1,ccoord[1,:]<min_angle] += 2*np.pi
    return ccoord

def sinoidModel(ccoord,components=4):
    phi = ccoord[1,:]
    z = ccoord[2,:]
    model = np.array([  z*0+1,
                        z,
                      ])
    for c in range(1,components+1):
        model = np.vstack((model,
                    np.array([
                        np.cos(c*phi),
                        np.sin(c*phi),
                        z*np.cos(c*phi),
                        z*np.sin(c*phi),
                    ])))
    return model.T

def fitModel(ccoord):
    model = sinoidModel(ccoord)
    r = ccoord[0,:]
    coeffs, _, _, _ = np.linalg.lstsq(model, r, rcond=None)
    return coeffs

def applyModel(ccoord,coeffs):
    model = sinoidModel(ccoord)
    r = np.matmul(model,coeffs)
    return r

@retry(3)
def ransacSinoidFit(ccoord,minDataPoints=50,iterations=10,inlierThreshold=.005,minInlierFraction=0.90,VERBOSE=False):
    N = ccoord.shape[1]
    besterr = np.inf
    bestinliers = [0]
    bestfit = None
    for iter in range(iterations):
        maybeinliers = np.random.randint(N, size=minDataPoints)
        maybemodel = fitModel(ccoord[:,maybeinliers])
        alsoinliers = np.abs(applyModel(ccoord,maybemodel)-ccoord[0,:]) < inlierThreshold
        thiserr = np.inf
        if sum(alsoinliers) > N*minInlierFraction:
            if VERBOSE:
                print(f"        ({iter+1:2d}/{iterations:2d}) First step inliers: {100*sum(alsoinliers)/N:.1f}%, ",end='',flush=True)
            bettermodel = fitModel(ccoord[:,alsoinliers])
            thiserr = np.mean(np.abs(applyModel(ccoord[:,alsoinliers], bettermodel)-ccoord[0,alsoinliers]))
            betterinliers = np.abs(applyModel(ccoord,bettermodel)-ccoord[0,:]) < inlierThreshold
            if  thiserr < besterr:
                bestinliers = np.where(betterinliers)[0]
                bestfit = bettermodel
                bestinliers = betterinliers
                besterr = thiserr
                if VERBOSE:
                    print(f"Mean error: {thiserr:.2E} *")
            else:
                if VERBOSE:
                    print(f"Mean error: {thiserr:.2E}")
        else:
            if VERBOSE:
                print(f"        ({iter+1:2d}/{iterations:2d}) First step inliers: {100*sum(alsoinliers)/N:.1f}%, not enough!")
    if bestfit is None:
        print("No appropriate model found!")
        raise Exception("RANSAC did not find an appropriate model!")
    inliers = np.zeros([N,],dtype=bool)
    inliers[bestinliers] = True
    return bestfit,inliers

def flattenSurface(ccoord,coeffs):
    flattened = ccoord.copy()
    r = applyModel(flattened,coeffs)
    flattened[0,:] -= r
    flattened[2,:] *= -1
    flattened[1,:] *= np.nanmean(r)
    return flattened[[1,2,0],:]

