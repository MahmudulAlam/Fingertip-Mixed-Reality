# Affine Transformation of Virtual Object
A convolutional neural network (CNN) based thumb and index fingertip detection and tracking system are presented here for seamless
interaction with a virtual 3D object in the virtual environment. First, a two-stage CNN is employed to detect the hand and fingertips 
and using the information of the fingertip position, the scale, rotation, translation, and in general affine transformation of virtual 
object is performed. 

## Update 
This is the version ```2.0``` that includes a more generalized affine transformation of virtual objects in the virtual environment with
more experimentation and analysis. Version [```1.0```](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/releases/tag/v1.0) 
only contains the geometric transformation of a virtual 3D object with respect to a finger gesture.

[![GitHub stars](https://img.shields.io/github/stars/MahmudulAlam/Fingertip-Mixed-Reality)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MahmudulAlam/Fingertip-Mixed-Reality)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/network)
[![Downloads](https://img.shields.io/badge/version-2.0-orange.svg?longCache=true&style=flat)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality)
[![GitHub license](https://img.shields.io/github/license/MahmudulAlam/Fingertip-Mixed-Reality)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/blob/master/LICENSE)

## Dataset
To train the hand and fingertip detection model two different datasets are used. One is self-made publicly released dataset called
[TI1K Dataset](https://github.com/MahmudulAlam/TI1K-Dataset) which contains 1000 images with the annotations of hand and fingertip position and another one is [Scut-Ego-Gesture Dataset](http://www.hcii-lab.net/data/SCUTEgoGesture/index.htm). 

## Requirements
- [x] TensorFlow-GPU==1.15.0
- [x] OpenCV==4.2.0
- [x] Cython==0.29.2
- [x] ImgAug==0.2.6
- [x] Weights: [```download```](https://www.dropbox.com/sh/2cwwbxpklglpr8l/AAC6kWnN7iJZHn5wNYO7VLTKa?dl=0) the trained weights file for both hand and fingertip detection model and put the weights folder in the working directory. 

[![Downloads](https://img.shields.io/badge/download-weights-blue.svg?style=popout-flat&logo=dropbox)](https://www.dropbox.com/sh/2cwwbxpklglpr8l/AAC6kWnN7iJZHn5wNYO7VLTKa?dl=0)

## System Overview
Here it the real-time demo of the scale, rotation, translation, and overall affine transformation of the virtual object using finger 
interaction.

<pre>
  <strong>                scale transformation                                  rotation transformation   </strong>
</pre>
<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/75994594-6881ff00-5f25-11ea-9194-ac44a082dcc6.gif" title="scale transformation" width="400" />
  <img src="https://user-images.githubusercontent.com/37298971/75994991-f958da80-5f25-11ea-9db3-0bc079f5e687.gif" title="rotation transformation" width="400" />
</p>
<pre>
  <strong>             translation transformation                                 affine transformation   </strong>
</pre>
<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/75995017-0249ac00-5f26-11ea-91ea-2a19a63384ca.gif" title="translation transformation" width="400" />
  <img src="https://user-images.githubusercontent.com/37298971/75995036-0675c980-5f26-11ea-967a-10b9fe45853d.gif" title="affine transformation"width="400" />
</p>

## Experimental Setup
The experimental setup has a server and client-side. Fingertip detection and tracking and all other machine learning stuff are 
programmed in the server-side using Python. In the client-side, the virtual environment is created using Unity along with Vuforia 
software development kit (SDK). To locate and track a virtual object using the webcam, Vuforia needs marker assistance. For that 
purpose, a marker is designed which works as an image target. The ```marker``` folder contains the pdf of the designed marker. To use
the system print a copy of the marker.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/57572552-c9dc5d00-743d-11e9-9f1c-fcf9d97517b5.jpg" width="800">
</p>

## How to Use
To run the server-side with tracker directly run ```'server_track.py'``` and for without tracker run ```'server.py'```
It will wait until the client-side (Unity) is starting to send images to the server. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/57572729-97802f00-7440-11e9-9c5c-fa4d6427d8ec.png" width="700">
</p>

Open the ```'Unity Mixed Reality With Finger Gesture'``` environment and hit play button. Make sure a webcam is connected. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/57572793-5b010300-7441-11e9-96cb-07bec8f818b1.png" width="700">
</p>

Bring your hand in front of the webcam and interact with the virtual object using your finger gesture.

<!---
## Demo
<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/60720908-07e19300-9f4e-11e9-8e38-f322d81a9abd.gif" width="700">
</p>
-->
