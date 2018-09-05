# Overview

A test implementation for the paper "IDHashGAN: Deep Hashing with Generative Adversarial Nets for Incomplete Data Retrieval" (Under reviewing)

# Environment: 
  python 3.6

# Supported Toolkits
  pytorch (Pytorch http://pytorch.org/)
  
  torchvision
  
  numpy
  
# Demo

  1. Download pre-trained models from [BaiduNetdisk](https://pan.baidu.com/s/1z8lhFlAr3_YthTrNywMNtw). password: amwa.

  2. Download dataset from [BaiduNetdisk](https://pan.baidu.com/s/1z8lhFlAr3_YthTrNywMNtw), then put all this data into corresponding dir and extract all compressed files.
  
     tar -xvf database_img.tar
     
     tar -xvf train_img.tar
     
     tar -xvf test_img.tar
     
     tar -xvf recon_img.tar
     
  3. Copy the model into your dir
  
     cp netG*.pth ./model/
     
     cp netF*.pth ./model/

  4. Test for the retrieval of incomplete data using the proposed IDHashGAN
  
     python test_for_idhash_IDHashGAN.py
     
  5. Test for the retrieval of incomplete data using the compared method
  
     python test_for_idhash_DPSH.py

  6. Test for the retrieval of complete data using the compared methods
  
     python test_for_cdhash_IDHashGAN.py

  7. Test for the restroation of incomplete data
  
     python test_for_visualization.py --netG ./model/netG_flickr.pth --test_image ./data/MIRFLICKR/test_img/2.png
     
     python test_for_visualization.py --netG ./model/netG_flickr.pth --test_image ./data/MIRFLICKR/test_img/36.png
     
     python test_for_visualization.py --netG ./model/netG_flickr.pth --test_image ./data/MIRFLICKR/test_img/40.png
        
# Notes
- This is developed on a Linux machine running Ubuntu 16.04.
- Use GPU for the high speed computation.
