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
from data_loader import ImagerLoader
import numpy as np
from videogaze_model import VideoGaze
import cv2
import math
import sklearn.metrics

#Path for file 
source_path = "images"
face_path = "faces"
target_path = "target"

#Train and test input files. Format is described in README.md 
test_file = "test_flag.txt"
train_file = "train_flag.txt"

#Training parameters
workers = 30;
epochs = 900
batch_size = 200

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 0
lr = base_lr

side_w = 20


#Define the exponential Shifted Grids Loss
class ExponentialShiftedGrids(nn.Module):
    def __init__(self):
        super(ExponentialShiftedGrids, self).__init__()
        self.filters = []
        self.sigmoid = nn.Softmax()
        for i in range(0,side_w):
            for j in range(0,side_w):
                f = nn.Conv2d(1, 1, kernel_size=(i+1,j+1), padding=(math.floor((i+1-1)/2),math.floor((j+1-1)/2)),bias=False)
                f.weight[0].data[:,:,:] =1
                self.filters.append(f.cuda())

    def forward(self, output_o,target_o):
        output_o = self.sigmoid(output_o.view(-1,side_w*side_w))
        output_o = output_o.view(-1,1,side_w,side_w)
        loss = 0

        for i in range(side_w):
            for j in range(side_w):
                filt = self.filters[i*side_w+j]
                output = filt(output_o)
                target = filt(target_o)
                x = torch.mul(output,target)
                x = torch.clamp(x,min=0.0001)
                x = -torch.log(x)
                x = torch.mul(x,target)
                loss = loss+torch.div(torch.mean(x),(i+1)*(j+1)/(side_w*side_w))
        return loss

def main():
    global args, best_prec1,weight_decay,momentum

    model = VideoGaze(batch_size)

    model.cuda()

    # optionally resume from a checkpoint

    cudnn.benchmark = True

    
    #Define training loader
    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(source_path,face_path,target_path,train_file,transforms.Compose([
            transforms.ToTensor(),
        ]),square=(227,227),side=side_w),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    #Define validation loader
    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(source_path,face_path,target_path,test_file,transforms.Compose([
            transforms.ToTensor(),
        ]),square=(227,227),side=side_w),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)



    #Define loss and optimizer
    criterion = ExponentialShiftedGrids().cuda()
    criterion_b = nn.BCELoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    #Training loop
    for epoch in range(0, epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion,criterion_b, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion,criterion_b)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion,criterion_b,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    topb = AverageMeter()
    l2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (source_frame,target_frame,face_frame,eyes,target_i,gaze_float,binary_tensor) in enumerate(train_loader):
        #Convert target values into grid format
        target = float2grid(target_i,side_w)
        target = target.float()

        # Prepare tensors and variables
        data_time.update(time.time() - end)
        source_frame = source_frame.cuda(async=True)
        target_frame = target_frame.cuda(async=True)
        face_frame = face_frame.cuda(async=True)
        eyes = eyes.cuda(async=True)
        target = target.cuda(async=True)
        binary_tensor = binary_tensor.cuda(async=True)
        target_i = target_i.cuda(async=True)


        source_frame_var = torch.autograd.Variable(source_frame)
        target_frame_var = torch.autograd.Variable(target_frame)
        face_frame_var = torch.autograd.Variable(face_frame)
        eyes_var = torch.autograd.Variable(eyes)
        target_var = torch.autograd.Variable(target)
        binary_var = torch.autograd.Variable(binary_tensor.view(-1))

        # compute output
        output,sigmoid= model(source_frame_var,target_frame_var,face_frame_var,eyes_var)

        #Compute loss
        loss_l2 = criterion(output, target_var)
        loss_b = criterion_b(sigmoid, binary_var)
        loss = loss_l2+12*loss_b

        # measure performance and record loss
        prec1, prec5 = accuracy(output.data, target_i.view(-1), topk=(1, 5))
        prec1_b = ap_b(sigmoid.data, binary_tensor, topk=(1,))
        l2e = l2_error(output, target_i.view(-1),gaze_float)
        losses.update(loss.data[0], source_frame.size(0))
        top1.update(prec1[0], source_frame.size(0))
        top5.update(prec5[0], source_frame.size(0))
        l2.update(l2e, source_frame.size(0))
        topb.update(prec1_b, source_frame.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) \t'
                  'Binary {topb.val:.3f} ({topb.avg:.3f}) \t'
                  'L2 {l2.val:.3f} ({l2.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5,l2=l2,topb=topb))

def validate(val_loader, model, criterion,criterion_b):
    global count_test
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    topb = AverageMeter()
    l2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (source_frame,target_frame,face_frame,eyes,target_i,gaze_float,binary_tensor) in enumerate(val_loader):
        #Convert target values into grid format
        target = float2grid(target_i,side_w)
        target = target.float()

        # Prepare tensors and variables
        source_frame = source_frame.cuda(async=True)
        target_frame = target_frame.cuda(async=True)
        face_frame = face_frame.cuda(async=True)
        eyes = eyes.cuda(async=True)
        target = target.cuda(async=True)
        binary_tensor = binary_tensor.cuda(async=True)
        target_i = target_i.cuda(async=True)


        source_frame_var = torch.autograd.Variable(source_frame)
        target_frame_var = torch.autograd.Variable(target_frame)
        face_frame_var = torch.autograd.Variable(face_frame)
        eyes_var = torch.autograd.Variable(eyes)
        target_var = torch.autograd.Variable(target)
        binary_var = torch.autograd.Variable(binary_tensor.view(-1))

        # compute output
        output,sigmoid= model(source_frame_var,target_frame_var,face_frame_var,eyes_var)

        #Compute loss
        loss_l2 = criterion(output, target_var)
        loss_b = criterion_b(sigmoid, binary_var)
        loss = loss_l2+12*loss_b

        # measure performance and record loss
        prec1, prec5 = accuracy(output.data, target_i.view(-1), topk=(1, 5))
        prec1_b = ap_b(sigmoid.data, binary_tensor, topk=(1,))
        l2e = l2_error(output, target_i.view(-1),gaze_float)
        losses.update(loss.data[0], source_frame.size(0))
        top1.update(prec1[0], source_frame.size(0))
        top5.update(prec5[0], source_frame.size(0))
        l2.update(l2e, source_frame.size(0))
        topb.update(prec1_b, source_frame.size(0))


        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) \t'
                  'Binary {topb.val:.3f} ({topb.avg:.3f}) \t'
                  'L2 {l2.val:.3f} ({l2.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5,l2=l2,topb=topb))

    return l2.avg


def save_checkpoint(state, is_best, filename='checkpoint_short.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_shape2.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def l2_error(output, target,target_float):
    """Computes the precision@k for the specified values of k"""
    maxk = 1
    batch_size = target.size(0)
    mean_sum = 0
    count=0
    target = target.cpu().numpy()
    output = output.data

    for i in range(batch_size):
        int_class = target[i]
        if int_class<side_w*side_w:
            map = output[i,:].view(side_w,side_w).cpu()
            map = map.numpy()
            map = cv2.resize(map,(200,200))
            map = np.reshape(map,(200*200))
            int_class = np.argmax(map)
            x_class = int_class % 200
            y_class = (int_class-x_class)//200
            y_float = y_class/200.0
            x_float = x_class/200.0
            mean_sum = mean_sum + np.sqrt((x_float-target_float[i,0,0])**2+(y_float-target_float[i,0,1])**2)
            count = count+1

    return mean_sum/count


def float2grid(input,side):
    b_s = input.size(0)
    output = torch.zeros(b_s,side*side)
    for j in range(b_s):
        class_final = input[j,0,0]
        if(class_final < side*side):
            output[j,class_final] =1
    output = output.view(-1,1,side,side)
    return output



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output = output.view(-1,side_w*side_w)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def ap_b(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    target = target.view(-1).cpu().numpy()
    output = output.view(-1).cpu().numpy()
    return sklearn.metrics.average_precision_score(target,output)



if __name__ == '__main__':
    main()

