# Endo-Depth-and-Motion-merge
The main code is from https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion

This repository merge the code of Endo-Depth's depth prediction from single images, the photometric and the others trackings methods and the volumetric fusion used in the paper

Now compelete merging the photometric part and the volumetric fusion part,now put an effort on merging the depth prediction from single images part

# About
Endo-Depth-and-Motion is a pipeline where first, pixel-wise depth is predicted on a set of keyframes of the endoscopic monocular video using a deep neural network (Endo-Depth). The motion of each frame with respect to the closest keyframe is estimated by minimizing the photometric error, robustified using image pyramids and robust error functions. Finally, the depth maps of the keyframes are fused in a Truncated Signed Distance Function (TSDF)-based volumetric representation.
# Setup
We have ran our experiments under CUDA Version 10.1.105, CuDNN 7.6.5 and Ubuntu 18.04. We recommend create a virtual environment with Python 3.6 using Anaconda `conda create -n edam python=3.6` and install the dependencies as
```=
conda install -c conda-forge opencv=4.2.0
pip3 install -r path/to/Endo-Depth-and-Motion-merge/requirements.txt
```
# Endo-Depth
To predict the depth for a single or multiple images use
```=
python apps/depth_estimate/__main__.py --image_path path/to/image_folder --model_path path/to/model_folder
```
# Tracking
You can execute the photometric tracking with
```=
python apps/tracking_ours/__main__.py -d cuda:0 -i path/to/hamlyn_tracking_test_data -o apps/tracking_ours/results
```
```
dataset_folder   
      -->test1      
         -->color	 
	 -->depth	       
	 -->intrinsics.txt	       
      ...
```
To use alternatively the tracking methods of Open3D run
```=
python apps/tracking_open3d/__main__.py -d cuda:0 -i path/to/hamlyn_tracking_test_data -o apps/tracking_open3d/results -t park
```
Tips for the visualization. When the two windows (images and 3D map) display, left click on the middle of the images window and then you can use the following commands pressing the buttons:
```=
a: to start the automode. The currently displayed scene will be tracked and viewed in real time in the 3D window.
s: to stop the automode. This can only be done when one frame is finally tracked and before the next one is started. So just smash the button multiple times until it stops!
h: to print help about more commands, like skip the scene or to track frame by frame.
```
# Volumetric fusion
```=
python apps/volumetric_fusion/__main__.py -i apps/tracking_ours/results/test1.pkl -o path/to/hamlyn_tracking_test_data/test1
```
# The merge of Endo-Depth and Tracking and Volumetric fusion
```=
python apps/tracking_ours/__newmain__.py -d cuda:0 -i Hamlyn_tracking_test_data -o Hamlyn_tracking_test_data/test1/ -m Stereo_loss_models/epoch1/
```

# hinet

```=
https://github.com/google-research/google-research/tree/master/hitnet?fbclid=IwAR0kR3lEabzoBXK2SKnoK_OOCTrQm1f1zZ3q3Rtg0AyK6cP5YMbXxYRC0u0
https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation?fbclid=IwAR2vm1AA-C9NWqk7DZ0eMh_hsRMjmQPLzuGeuY93k3rrWZHHzADi1yBpWYg
```
