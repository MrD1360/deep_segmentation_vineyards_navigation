
# Semantic Segmentation Controller for Vineyards
A control system that exploits semantic segmentation properties to properly drive the mobile platform along vineyard rows.

## Content of repository
- _Model\_training\_and\_validation\_\_segmentation.ipynb_ is a jupyter notebook used for training and model validation.
- _model\_mobile\_seg\_fp32.tflite_ is the obtained model weights optimized for running on CPU.
- _ROSController.py_ is a ROS implementation of the controller.


## Test dataset
Test dataset available at: 10.5281/zenodo.4601472


## Dependencies and libraries
The system has been tested on tensorflow==2.4.1 version and ROS Melodic.

