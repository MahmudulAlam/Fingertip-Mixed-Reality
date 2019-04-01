# Fingertip-Mixed-Reality
In this project, a thumb and index fingertip detection and tracking system for seamless interaction with a virtual 3D object in the 
MR environment is developed. First, a two stage convolutional neural networks (CNN) is used to detect hand and fingertip and using the 
information of the fingertip position, the scale of a virtual 3D object is controlled. 

## Dataset
To train the hand and fingertip detection model two different datasets are used. One is self-made publicly released dataset called
[TI1K Dataset](https://github.com/MahmudulAlam/TI1K-Dataset) and another one is [Scut-Ego-Gesture Dataset](http://www.hcii-lab.net/data/SCUTEgoGesture/index.htm). 

[![Download](https://img.shields.io/badge/download-dataset-blueviolet.svg?longCache=true&style=flat)](https://github.com/MahmudulAlam/TI1K-Dataset/archive/master.zip)

## Requirements
- [x] TensorFlow-GPU==1.11.0
- [x] NumPy==1.15.4
- [x] OpenCV==3.4.4
- [x] Cython==0.29.2
- [x] ImgAug==0.2.6
- [x] Weights: Download the trained weights file for both hand and fingertip detection model and put the weights folder in the working directory. 

[![Downloads](https://img.shields.io/badge/download-weights-orange.svg?longCache=true&style=flat)](https://mega.nz/#F!LssiECRa!wrI3o59ccLbNYvDOZHm1ow)

## System-Overview
Here is a brief overview of the system where a virtual 3D object is scaled up in (a) and down in (b). 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/54619957-9dd0db00-4a8f-11e9-9a83-18b0d9ddfa4f.png" width="700">
</p>
