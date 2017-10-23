Author: Adrià Recasens (recasens@mit.edu)

In this folder, you will find the essential files for training and using the video gazefollowing model.
# Data #
The VideoGaze dataset can be found [here](http://videogazefollow.csail.mit.edu/downloads/data.zip). 
# Training #

In train.py, you will find the basic training file. The training data is specified in train.txt and test.txt. The format for this files is:

source_name target_name face_name flip eyes_x eyes_y gaze_x gaze_y


where 
- source_name is the relative path to the source frame
- target_name is the relative path to the target frame
- face_name is the relative path to the cropped face image
- flip means wether the images should be flipped. The gaze will need to be flipped beforehand.
- eyes_x and eyes_y is the eyes location (assuming the image is 1x1, as provided in the data) 
- gaze_x and gaze_y is the gaze location. If gaze is a negative value, this means the gaze is not present in the target frame. 

The path for the faces, source frame and target frame images is also described in the begging of the training file. 

# Demo #
In demo.py you will find the basic usage of the model. This file loads the model, a video and a picture of a character in the video. It wil track the character and detect where the person is looking trhough the video. 


# Requirements #
1. [PyTorch](pytorch.org)
2. [face_recognitio](https://pypi.python.org/pypi/face_recognition)

