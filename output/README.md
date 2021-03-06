These directories will contain output files after running the demo.

.npy files can be opened with numpy.load(filename) in python.

## 00_reference.jpg

The unchanged reference image.

## 01_disparity.jpg

The disparity in grayscale ranging from minimum (20px) in black to maximum (220px) in white.  
Magenta pixels indicate no match found for that pixel.

## 02_pointcloud.npy

The reconstructed point cloud.  
Rows correspond to points, Columns correspond to:  
[X,Y,Z,anomalyscore,R,G,B]  
where RGB are the color values of the reference image

## 03_fit.npy

Pointcloud of the difference between the fit and the cylindrically transformed data.  
Rows correspond to points, Columns correspond to:  
[anomalyscore,phi\*r0,Z,R,G,B]  
where phi\*r0 is the angle scaled with the average radius, to be in proportion with Z.

## 04_highlighted.jpg

The reference image with detected anomalies highlighted.  
Red pixels indicate anomalies, blue pixels indicate a failed stereomatch for that pixel or a point outside the valid Z-range.
