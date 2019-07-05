# Fingertip-Mixed-Reality
In this project, a thumb and index fingertip detection and tracking system for seamless interaction with a virtual 3D object in the 
MR environment is developed. First, a two-stage convolutional neural network (CNN) is used to detect hand and fingertip and using the 
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

## Experimental-Setup
The experimental setup has a server and client side. Fingertip detection and tracking and all other machine learning stuff are programmed in the server side using Python. In the client side, the virtual environment is created using Unity along with Vuforia software development kit (SDK). To locate and track a virtual object using the webcam, Vuforia needs marker assistance. For that purpose, a marker is designed which works
as an image target. The ```marker``` folder contains the pdf of the designed marker. To use the system print a copy of the marker.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/57572552-c9dc5d00-743d-11e9-9f1c-fcf9d97517b5.jpg" width="800">
</p>

## How to Use
To run the server side with tracker directly run ```'server_track.py'``` and for without tracker run ```'server.py'```
It will wait until the client side (Unity) is starting to send mages to the server. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/57572729-97802f00-7440-11e9-9c5c-fa4d6427d8ec.png" width="700">
</p>

Open the ```'Unity Mixed Reality With Finger Gesture'``` environment and hit play button. Make sure a webcam is connected. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/57572793-5b010300-7441-11e9-96cb-07bec8f818b1.png" width="700">
</p>

Bring your hand in front of the webcam and interact with the virtual object using your finger gesture.

## Demo
<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/60720908-07e19300-9f4e-11e9-8e38-f322d81a9abd.gif" width="650">
</p>
