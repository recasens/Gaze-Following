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
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable



model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256,13,13)
        return x

class HeadPoseAlexnet(nn.Module):

    def __init__(self, num_classes=1000):

        super(HeadPoseAlexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class HeadPose(nn.Module):

    def __init__(self, num_classes=1000):
        super(HeadPose, self).__init__()
        self.alexnet = HeadPoseAlexnet()
        self.alexnet.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        self.linear1 = nn.Linear(256*6*6,500)
        self.threshold1 = nn.Threshold(0, 1e-6)
        self.linear2 = nn.Linear(500,200)
        self.threshold2 = nn.Threshold(0, 1e-6)
        self.linear3 = nn.Linear(200,4)

    def forward(self, x):
        x = self.alexnet(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.linear1(x)
        x = self.threshold1(x)
        x = self.linear2(x)
        x = self.threshold2(x)
        x = self.linear3(x)
        x = x.view(x.size(0), 4)
        return x




class TransformationPathway(nn.Module):

    def __init__(self, num_classes=1000):
        super(TransformationPathway, self).__init__()
        self.alexnet = HeadPoseAlexnet()
        self.alexnet.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

        self.conv_features = nn.Sequential(
                      nn.Conv2d(512,100,kernel_size=3,padding=1,stride=1),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=3, stride=2))
        self.linear_features = nn.Sequential(
                      nn.Linear(400, 200),
                      nn.ReLU(inplace=True),
                      nn.Linear(200, 100),
                      nn.ReLU(inplace=True)
            )
        self.final_linear = nn.Linear(100,7)
        self.final_linear.bias.data[0] = 0
        self.final_linear.bias.data[1] = 0
        self.final_linear.bias.data[2] = 0.3
        self.final_linear.bias.data[3] = 0
        self.final_linear.bias.data[4] = 0
        self.final_linear.bias.data[5] = 0
        self.final_linear.bias.data[6] = 0.4


    def forward(self, source,target):
        source_conv = self.alexnet(source)
        target_conv = self.alexnet(target)
        all_conv = torch.cat((source_conv,target_conv),1)
        conv_output = self.conv_features(all_conv).view(-1,400)
        fc_output = self.linear_features(conv_output)
        fc_output = self.final_linear(fc_output)
        angles = torch.mul(nn.Tanh()(fc_output[:,3:6]),np.pi)
        R = self.rotation_tensor(angles[:,0],angles[:,1] , angles[:,2], angles.size(0))
        x_t = fc_output[:,0:3]
        sigmoid = nn.Hardtanh(0,1)(fc_output[:,6])
        return R,x_t,sigmoid

    def rotation_tensor(self,theta, phi, psi, n_comps):
        rot_x = Variable(torch.zeros(n_comps, 3, 3).cuda())
        rot_y = Variable(torch.zeros(n_comps, 3, 3).cuda())
        rot_z = Variable(torch.zeros(n_comps, 3, 3).cuda())
        
        rot_x[:, 0, 0] = 1
        rot_x[:, 1, 1] = theta.cos()
        rot_x[:, 1, 2] = theta.sin()
        rot_x[:, 2, 1] = -theta.sin()
        rot_x[:, 2, 2] = theta.cos()
    
        rot_y[:, 0, 0] = phi.cos()
        rot_y[:, 0, 2] = -phi.sin()
        rot_y[:, 1, 1] = 1
        rot_y[:, 2, 0] = phi.sin()
        rot_y[:, 2, 2] = phi.cos()
    
        rot_z[:, 0, 0] = psi.cos()
        rot_z[:, 0, 1] = -psi.sin()
        rot_z[:, 1, 0] = psi.sin()
        rot_z[:, 1, 1] = psi.cos()
        rot_z[:, 2, 2] = 1
        rot_2 = torch.bmm(rot_y, rot_x)
        return torch.bmm(rot_z, rot_2)
        

class ConeProjection(nn.Module):

    def __init__(self, batch_size=100):
        super(ConeProjection, self).__init__()
        self.batch_size = batch_size


    def forward(self,eyes,v,R,t,alpha):

        P = Variable(torch.zeros(eyes.size(0), 169, 3).cuda())
        for b in range(eyes.size(0)):
            for i in range(13):
                for j in range(13):
                    k = 13*i + j 
                    P[b,k,0] = (i-6)/6
                    P[b,k,1] = (j-6)/6
                    P[b,k,2] = 1


        id_matrix = Variable(torch.zeros(eyes.size(0), 3, 3).cuda())
        id_matrix[:,0,0] = alpha
        id_matrix[:,1,1] = alpha
        id_matrix[:,2,2] = alpha

        #Normalize vector!
        v = v / v.norm(2, 1).clamp(min=0.00000000000001).view(-1,1).expand_as(v)


        v_matrix = torch.bmm(v.view(-1,3,1),v.view(-1,1,3))
        

        M = v_matrix-id_matrix

        sigma_matrix = Variable(torch.zeros(eyes.size(0), 3, 3).cuda())

        v1 = R[:,:,0].contiguous().view(-1,3)
        v2 = R[:,:,1].contiguous().view(-1,3)

        u_e = eyes

        v11 = v1.contiguous().view(-1,1,3)
        v21 = v2.contiguous().view(-1,1,3)
        v12 = v1.contiguous().view(-1,3,1)
        v22 = v2.contiguous().view(-1,3,1)
        u_e1 = u_e.contiguous().view(-1,1,3)
        u_e2 = u_e.contiguous().view(-1,3,1)
        t1 = t.contiguous().view(-1,1,3)
        t2 = t.contiguous().view(-1,3,1)


        sigma_matrix[:,0:1,0:1] = torch.bmm(v11,torch.bmm(M,v12))
        sigma_matrix[:,0:1,1:2] = torch.bmm(v11,torch.bmm(M,v22))
        sigma_matrix[:,0:1,2:3] = torch.bmm(v11,torch.bmm(M,(t2-u_e2)))
        sigma_matrix[:,1:2,0:1] = torch.bmm(v21,torch.bmm(M,v12))
        sigma_matrix[:,1:2,1:2] = torch.bmm(v21,torch.bmm(M,v22))
        sigma_matrix[:,1:2,2:3] = torch.bmm(v21,torch.bmm(M,t2-u_e2))
        sigma_matrix[:,2:3,0:1] = torch.bmm(t1-u_e1,torch.bmm(M,v12))
        sigma_matrix[:,2:3,1:2] = torch.bmm(t1-u_e1,torch.bmm(M,v22))
        sigma_matrix[:,2:3,2:3] = torch.bmm(t1-u_e1,torch.bmm(M,t2-u_e2))

        sigma_matrix_all = sigma_matrix.view(-1,1,3,3).expand(eyes.size(0),169,3,3).contiguous().view(-1,3,3)
        P1 = P.contiguous().view(-1,1,3)
        P2 = P.contiguous().view(-1,3,1)
        sum_all = torch.bmm(P1,torch.bmm(sigma_matrix_all,P2)).contiguous().view(-1,169)

        return sum_all


class VideoGaze(nn.Module):


    def __init__(self, bs=200,side=20):
        super(VideoGaze, self).__init__()
        self.saliency_pathway = AlexNet()
        self.saliency_pathway.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        self.last_conv = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.relu_saliency = nn.ReLU(inplace=True)
        self.cone_pathway = HeadPose()
        self.projection = ConeProjection(bs)
        self.transformation_path = TransformationPathway()
        self.linear_final = nn.Linear(169,side*side)
        self.sigmoid1 = nn.Linear(169*2,200)
        self.sigmoid2 = nn.Linear(200,1)
        self.last_convolution = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.side = side

    def forward(self, source,target,face,eyes):
        saliency_256 = self.saliency_pathway(target)
        saliency_output = self.last_conv(saliency_256)
        saliency_output = self.relu_saliency(saliency_output)
        saliency_output = saliency_output.view(-1,169)
        cone_parameters = self.cone_pathway(face)
        head_v = cone_parameters[:,0:3]
        variance = nn.Hardtanh(0.5, 0.99)(cone_parameters[:,3])
        R,t,sigmoid = self.transformation_path(source,target)
        projection = self.projection(eyes,head_v,R,t,variance)
        projection_simoid = torch.mul(projection,sigmoid.view(-1,1).expand_as(projection))
        
        input_sigmoid = torch.cat((saliency_output,projection_simoid),1)
        output_sigmoid_l1 = nn.ReLU()(self.sigmoid1(input_sigmoid))
        output_sigmoid_l2 = self.sigmoid2(output_sigmoid_l1)
        output_sigmoid_l2 = nn.Sigmoid()(output_sigmoid_l2)

        output = torch.mul(projection_simoid,saliency_output)
        output = self.linear_final(output)
        output = nn.Softmax()(output)
        output = output.view(-1,1,self.side,self.side)

        return output,output_sigmoid_l2

