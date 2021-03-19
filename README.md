# StereoDemo
Demo accompanying the article

Requires python3.6 or newer.

Install necessary packages:

    python3 -m pip install -r requirements.txt --user

Run the demo:

    python3 demo.py

After choosing a data set to run on, the script will run the first three steps of the framework, "Data Acquisition", "Semi-Global Stereo Matching", and "Three-Dimensional Geometry Reconstruction".

At this point the user will need to select a valid Z-range in the pptk pointcloud viewer, using instructions as specified in the terminal.
It's important to select a wide enough range to capture the overal shape of the pipe but not include too much points outside the pipe. Values around -1.5 and -2.0 are suggested for most image sets.

Output files will be written to the "output/" folder for closer inspection.

To navigate the pptk point cloud viewer, use:

* Click and drag to rotate
* `Shift`-click and drag to pan
* Scroll to zoom
* `1`, `3`, `7` keys to align the view with the X, Y, Z axes respectively
* `5` key to switch between perspective and orthographic view
* `[` and `]` keys to change the color attribute between anomaly score and rgb pixel color (only after model fit)
