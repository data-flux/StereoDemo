This directory will contain some temporary files after running the demo.

## 00_reference.jpg

The unchanged reference image.

## 01_highlighted.jpg

The reference image with detected anomalies highlighted.

## 02_pointcloud.npy

The reconstructed point cloud.  
Rows correspond to points, Columns correspond to:  
[X,Y,Z,anomalyscore,R,G,B]  
where RGB are the color values of the reference image

## 03_fit.npy

Pointcloud of the difference between the fit and the cylindrically transformed data.  
Rows correspond to points, Columns correspond to:  
[anomalyscore,phi*r0,Z,R,G,B]  
where phi*r0 is the angle scaled with the average radius, to be in proportion with Z.
