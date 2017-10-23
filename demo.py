import face_recognition
import cv2
import pylab
import imageio

import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from videogaze_model import VideoGaze
import cv2
import math
from sklearn import metrics


#Loading the model
model = VideoGaze(bs=batch_size,side=20)
checkpoint = torch.load('model.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
cudnn.benchmark = True


#Reading input video
video_name = 'video_test.mp4'
vid = imageio.get_reader(video_name,  'ffmpeg')
fps = vid.get_meta_data()['fps']
frame_list = []
for i, im in enumerate(vid):
    frame_list.append(im)

print('Frame List Created')


#Loading the features for tracking
p_image = face_recognition.load_image_file("face.jpg")
p_encoding = face_recognition.face_encodings(p_image)[0]


trans = transforms.ToTensor()

#Reading video
vid = imageio.get_reader(video_name,  'ffmpeg')
fps = vid.get_meta_data()['fps']

#Output video
target_writer = imageio.get_writer('output.mp4', fps=fps)




#N corresponds to the number of frames in the window to explore
N = 25

#w_T corresponds to the number of frames to skip when sampling the target window
w_T = 40

target_frame = torch.FloatTensor(N,3,227,227)
target_frame = target_frame.cuda()

face_frame = torch.FloatTensor(N,3,227,227)
face_frame = target_frame.cuda()

eyes = torch.zeros(N,3)
eyes = eyes.cuda()




for i in range(len(frame_list)):
    print('Processing of frame %d out of %d' % (i,len(frame_list)))

    #Avoid the problems with the video limit
    if i>w_fps*(N-1)//2 and i<(len(frame_list)-w_fps*(N-1)//2):
        
        #Reading the image 
        top=False
        im = frame_list[i]
        h,w,c = im.shape


        #Detecting the person inside the image
        tmp_encodings = face_recognition.face_encodings(im)
        results = face_recognition.compare_faces(tmp_encodings, p_encoding)
        face_locations = face_recognition.face_locations(im)
        for id,face_local in enumerate(face_locations):
            if results[id]==True:
                (top, right, bottom, left) = face_local



        #If detection, run the model
        if top:

            #Crop Face Image 
            crop_img = im[top:bottom, left:right]
            crop_img = cv2.resize(im,(227,227)) 

            #Resize Image   
            im = cv2.resize(im,(227,227))

            #Compute the center of the head and estimate the eyes location
            eyes[:,0] = (right+left)/(2*w)
            eyes[:,1] = (top+bottom)/(2*h)

            #Fill the tensors for the exploring window. Face and source frame are the same
            source_frame = trans(im).view(1,3,227,227)
            face_frame = trans(crop_img).view(1,3,227,227)
            for j in range(N-1):
                trans_im = trans(im).view(1,3,227,227)
                source_frame = torch.cat((source_frame,trans_im),0)
                crop_im = trans(crop_img).view(1,3,227,227)
                face_frame = torch.cat((face_frame,crop_im),0)

            #Fill the targets for the exploring window. 
            for j in range(N):
                target_im = frame_list[i+w_fps*(j-((N-1)//2))]
                target_im = cv2.resize(target_im,(227,227))
                target_im = trans(target_im)
                target_frame[j,:,:,:] = target_im

            #Run the model
            source_frame = source_frame.cuda(async=True)
            target_frame = target_frame.cuda(async=True)
            face_frame = face_frame.cuda(async=True)
            eyes = eyes.cuda(async=True)
            source_frame_var = torch.autograd.Variable(source_frame)
            target_frame_var = torch.autograd.Variable(target_frame)
            face_frame_var = torch.autograd.Variable(face_frame)
            eyes_var = torch.autograd.Variable(eyes)
            output,sigmoid= model(source_frame_var,target_frame_var,face_frame_var,eyes_var)
            

            #Recover the data from the variables
            sigmoid = sigmoid.data
            output = output.data

            #Pick the maximum value for the frame selection
            v,ids = torch.sort(sigmoid, dim=0, descending=True) 
            index_target = ids[0,0]

            #Pick the frames corresponding to the maximum value
            target_im = frame_list[i+w_fps*(index_target-((N-1)//2))].copy()
            output_target = cv2.resize(output[index_target,:,:,:].view(20,20).cpu().numpy(),(200,200))

            #Compute the gaze location

            map = np.reshape(output_target,(200*200))

            int_class = np.argmax(map)
            x_class = int_class % 200
            y_class = (int_class-x_class)//200
            y_float = y_class/200.0
            x_float = x_class/200.0
            x_point = math.floor(x_float*w)
            y_point = math.floor(y_float*h)


            #Prepare video output
            tim = cv2.circle(target_im,(x_point,y_point), 30, (255,0,0), -1)
            face_im = cv2.rectangle(frame_list[i].copy(), (left, top), (right, bottom), (0, 0, 255), 3)
            final_im = np.concatenate((face_im,tim),axis=1)
            target_writer.append_data(final_im)


target_writer.close()






