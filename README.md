# Affine Transformation of Virtual Object
A convolutional neural network (CNN) based thumb and index fingertip detection system are presented here for seamless interaction with a virtual 3D object in the virtual environment. First, a two-stage CNN is employed to detect the hand and fingertips, and using the information of the fingertip position, the scale, rotation, translation, and in general, the affine transformation of the virtual object is performed.

## Update 
This is the version ```2.0``` that includes a more generalized affine transformation of virtual objects in the virtual environment with more experimentation and analysis. Previous versions only include the geometric transformation of a virtual 3D object with respect to a finger gesture. To get the previous version visit [here](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/releases).

[![GitHub stars](https://img.shields.io/github/stars/MahmudulAlam/Fingertip-Mixed-Reality)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MahmudulAlam/Fingertip-Mixed-Reality)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/network)
[![Downloads](https://img.shields.io/badge/version-2.0-orange.svg?longCache=true&style=flat)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality)
[![GitHub license](https://img.shields.io/github/license/MahmudulAlam/Fingertip-Mixed-Reality)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/blob/master/LICENSE)

## Paper
Paper for the affine transformation of the virtual 3D object has been published in Virtual Reality & Intelligent Hardware, Elsevier Science Publishers in 2020. To get more detail, please go through the [```paper```](https://doi.org/10.1016/j.vrih.2020.10.001). Paper for the geometric transformation of the virtual object [```v1.0```](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/releases/tag/v1.0) has also been published. For more detail, please go through this [```paper```](https://ieeexplore.ieee.org/abstract/document/9035256). If you use the code or data from the project, please cite the following papers: 


| [![Paper](https://img.shields.io/badge/paper-ScienceDirect-f2862e.svg?longCache=true&style=flat)](https://doi.org/10.1016/j.vrih.2020.10.001) 	|   ![](https://img.shields.io/badge/-v2.0-brightgreen)	|
|:-:	|:-:	|

```
@article{alam2020affine,
  title={Affine transformation of virtual 3D object using 2D localization of fingertips},
  author={Alam, Mohammad Mahmudul and Rahman, SM Mahbubur},
  journal={Virtual Reality \& Intelligent Hardware},
  volume={2},
  number={6},
  pages={534--555},
  year={2020},
  publisher={Elsevier}
}
```

|  [![Paper](https://img.shields.io/badge/paper-IeeeXplore-blue.svg?longCache=true&style=flat)](https://ieeexplore.ieee.org/abstract/document/9035256) 	|  ![](https://img.shields.io/badge/-v1.0-brightgreen) 	|
|:-:	|:-:	|

```
@inproceedings{alam2019detection,
  title={Detection and Tracking of Fingertips for Geometric Transformation of Objects in Virtual Environment},
  author={Alam, Mohammad Mahmudul and Rahman, SM Mahbubur},
  booktitle={2019 IEEE/ACS 16th International Conference on Computer Systems and Applications (AICCSA)},
  pages={1--8},
  year={2019},
  organization={IEEE}
}
```

## System Overview
Here it the real-time demo of the scale, rotation, translation, and overall affine transformation of the virtual object using finger 
interaction.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/78501859-96a26b00-777f-11ea-9f33-977ea8feda09.gif" width="640">
</p>

| <img src="https://user-images.githubusercontent.com/37298971/82154840-6d9eeb00-9892-11ea-986d-6314a21d5283.gif">  | <img src="https://user-images.githubusercontent.com/37298971/82154849-7abbda00-9892-11ea-84e9-4e1b5508c1e8.gif">  | <img src="https://user-images.githubusercontent.com/37298971/82154850-81e2e800-9892-11ea-8d7f-7d6830978a49.gif">  |
|:-:|:-:|:-:|

## Dataset
To train the hand and fingertip detection model two different datasets are used. One is a self-made publicly released dataset called [TI1K Dataset](https://github.com/MahmudulAlam/TI1K-Dataset) which contains 1000 images with the annotations of hand and fingertip position and another one is [Scut-Ego-Gesture Dataset](http://www.hcii-lab.net/data/SCUTEgoGesture/index.htm). 

## Requirements
- [x] TensorFlow-GPU==1.15.0
- [x] OpenCV==4.2.0
- [x] Cython==0.29.2
- [x] ImgAug==0.2.6
- [x] Weights: [```download```](https://mega.nz/folder/SkklxRAA#Z0p60mPe1BwJ7ZTZniMtDA) the trained weights file for both hand and fingertip detection model and put the weights folder in the working directory. 

[![Downloads](https://img.shields.io/badge/download-weights-blue.svg?style=popout-flat&logo=mega)](https://mega.nz/folder/SkklxRAA#Z0p60mPe1BwJ7ZTZniMtDA)

## Experimental Setup
The experimental setup has a server and client-side. Fingertip detection and tracking and all other machine learning stuff are programmed in the server-side using Python. On the client-side, the virtual environment is created using Unity along with the Vuforia software development kit (SDK). To locate and track a virtual object using the webcam, Vuforia needs marker assistance. For that purpose, a marker is designed which works as an image target. The [```marker/```](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/tree/master/marker) folder contains the pdf of the designed marker. To use the system print a copy of the marker.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/57572552-c9dc5d00-743d-11e9-9f1c-fcf9d97517b5.jpg" width="800">
</p>

## How to Use
First, to run the server-side directly run ```'server.py'```. It will wait until the client-side (Unity) is starting to send images to the server. 
```
directory > python server_track.py
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/57572729-97802f00-7440-11e9-9c5c-fa4d6427d8ec.png" width="700">
</p>

Open the ```'Unity Affine Transformation'``` environment using [```Unity```](https://unity3d.com/get-unity/download) and hit the play button. Make sure a webcam is connected. 

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
