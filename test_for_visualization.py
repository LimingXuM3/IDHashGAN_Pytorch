#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat May 19 20:55:40 2018

@author: Bryan

Output the restored Samples

'''

import argparse
import random
import torch

import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import load_functions as L_F

from torch.autograd import Variable
from IDHashGAN_models import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='MIRFLICKR', help='cifar10 | nus-wide | mirflickr-25k')
parser.add_argument('--test_image', default='', help='path to dataset')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nBottleneck', type=int, default=4000, help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int, default=4, help='overlapping edges')
parser.add_argument('--nef',type=int, default=64, help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float, default=0.999, help='0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)

netG = Generator(opt)
netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
netG.eval()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

image = L_F.load_test_image(opt.test_image, opt.imageSize)
image = transform(image)
image = image.repeat(1, 1, 1, 1)

input_real = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
real_center = torch.FloatTensor(1, 3, int(opt.imageSize/4), int(opt.imageSize/4))

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
real_center = Variable(real_center)

input_real.data.resize_(image.size()).copy_(image)
input_cropped.data.resize_(image.size()).copy_(image)
#point = random.randint(0, int(opt.imageSize * 3/4))
point = int(opt.imageSize * 3/8)
real_center_cpu = image[:,:, point: point + int(opt.imageSize/4), point: point + int(opt.imageSize/4)]
real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

input_cropped.data[:,0, point: point + int(opt.imageSize/4), point: point + int(opt.imageSize/4)] = 2*117.0/255.0 - 1.0
input_cropped.data[:,1, point: point + int(opt.imageSize/4), point: point + int(opt.imageSize/4)] = 2*104.0/255.0 - 1.0
input_cropped.data[:,2, point: point + int(opt.imageSize/4), point: point + int(opt.imageSize/4)] = 2*123.0/255.0 - 1.0

fake = netG(input_cropped)

recon_image = input_cropped.clone()
recon_image.data[:,:,point: point + int(opt.imageSize/4), point: point + int(opt.imageSize/4)] = fake.data

L_F.save_image('restoration/real_samples.png',image[0])
L_F.save_image('restoration/cropped_samples.png',input_cropped.data[0])
L_F.save_image('restoration/recon_samples.png',recon_image.data[0])
