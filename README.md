# Endo-Depth-and-Motion-merge
the main code is from https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion

This repository merge the code of Endo-Depth's depth prediction from single images, the photometric and the others trackings methods and the volumetric fusion used in the paper

now compelete merging the photometric part and the volumetric fusion part,now put an effort on merging the depth prediction from single images part

# Setup
We have ran our experiments under CUDA Version 10.1.105, CuDNN 7.6.5 and Ubuntu 18.04. We recommend create a virtual environment with Python 3.6 using Anaconda conda create -n edam python=3.6 and install the dependencies as
```=
conda install -c conda-forge opencv=4.2.0
pip3 install -r path/to/Endo-Depth-and-Motion-merge/requirements.txt
```
# Endo-Depth
# Tracking
'
dataset_folder   
      -->rectified01      
         -->color	 
	 -->depth	       
	 -->intrinsics.txt	       
      ...
'
# Volumetric fusion
