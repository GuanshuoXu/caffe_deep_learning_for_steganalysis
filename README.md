# Deep CNN for JPEG steganalysis

Source code to reproduce the paper accepted by IH&MMSEC2017.

The DCT kernels are saved in /kernels. The directory to access them are hard-coded in /include/caffe/filler.hpp
. So before building, please change the directories to make sure the DCT kernels can be found. 

# Examples


# Citation

Please cite the following paper if the code helps your research.

Please also cite two additional papers if you are using the BN-RELU (bn_conv and relu_recover) combo.
