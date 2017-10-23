import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(source_path, face_path, target_path,file_name):
    images = []
    print(file_name)
    with open(file_name, 'r') as f:
        for line in f:
            line = line[:-1]
            line = line.replace("\t", " ")
            line = line.replace("  ", " ")
            split_lines = line.split(" ")
            if(len(split_lines)>3):
                path_source = '{0}/{1}'.format(source_path, split_lines[0])
                path_face = '{0}/{1}'.format(face_path, split_lines[2])
                path_target = '{0}/{1}'.format(target_path, split_lines[1])
                flip = int(float(split_lines[3]))

                eyes = np.zeros((3))
                gaze = np.zeros((2))

                eyes[0] = float(split_lines[4])
                eyes[1] = float(split_lines[5])
           
                gaze[0] = float(split_lines[6])
                gaze[1] = float(split_lines[7])

                item = (path_source,path_face,path_target,flip,eyes,gaze)
                images.append(item)
    return images


def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")


class ImagerLoader(data.Dataset):
    def __init__(self, source_path, face_path, target_path,file_name,
                transform=None, target_transform=None, loader=default_loader,square=(227,227),side=5):

        imgs = make_dataset(source_path, face_path, target_path,file_name)

        self.source_path = source_path
        self.face_path = face_path
        self.target_path = target_path
        self.file_name = file_name
        self.square = square


        self.imgs = imgs
        self.transform = transform
        self.target_transform = transform
        self.loader = loader
        self.side = side

    def __getitem__(self, index):
        path_source,path_face,path_target,flip,eyes,gaze = self.imgs[index]
        
        eyes_tensor = torch.Tensor(eyes)
        gaze_tensor = np.zeros((1,1)).astype(int)
        gaze_float = np.zeros((1,2)).astype(float)
        binary_tensor = np.zeros((1)).astype(float)

        gaze_float[0,0] = gaze[0]
        gaze_float[0,1] = gaze[1]

        x = np.rint(np.floor(gaze[0]*self.side))
        y = np.rint(np.floor(gaze[1]*self.side))

        if(x<0 or x>=self.side or y<0 or y>=self.side):
            gaze_tensor[0,0] = np.rint(self.side*self.side)
            binary_tensor[0] = 0
        else:
            gaze_tensor[0,0] = np.rint(y*self.side+x)
            binary_tensor[0] = 1

        gaze_tensor = torch.LongTensor(gaze_tensor)
        gaze_float = torch.FloatTensor(gaze_float)
        binary_tensor = torch.FloatTensor(binary_tensor)

        source = self.loader(path_source)
        face = self.loader(path_face)
        target_frame = self.loader(path_target)

        if flip==1:
            source = source.transpose(Image.FLIP_LEFT_RIGHT)
            face = face.transpose(Image.FLIP_LEFT_RIGHT)
            target_frame = target_frame.transpose(Image.FLIP_LEFT_RIGHT)

        if self.square:
            source = source.resize(self.square)
            face = face.resize(self.square)
            target_frame = target_frame.resize(self.square)
        if self.transform:
            source = self.transform(source)
            face = self.transform(face)
            target_frame = self.transform(target_frame)


        return source,target_frame,face,eyes_tensor,gaze_tensor,gaze_float,binary_tensor

        
    def __len__(self):
        return len(self.imgs)
