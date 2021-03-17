# StereoDemo
Demo accompanying the article

Requires python3.6 or newer.

Install necessary packages:

    python3 -m pip install -r requirements.txt --user

Run the demo:

    python3 demo.py

After choosing a data set to run on, the script will run the first three steps of the framework, "Data Acquisition", "Semi-Global Stereo Matching", and "Three-Dimensional Geometry Reconstruction".
At this point the user will need to select a valid Z-range in the pptk pointcloud viewer, using instructions as specified in the terminal.

Temporary output files will be written to the "output/" folder for closer inspection.
