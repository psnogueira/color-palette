# Color Palette
__color-palette__ is python program that utilizes a camera to detect and analyze your color palette.\
When the program detects your face, __press 'c'__ to capture your face and start the parsing.

It generates an image with your __face__, one of the 4 predefined __color Palettes__ and the __best__ 
__products__ to match your palette.

##  How to run it:
1. Connect a camera to your computer.\
*If you have more than 1 camera connected, change the camera index in "cv2.VideoCapture(0)" (line 148)
       
3. Run run.py\
   python run.py

## Requirements:
    pip install torch
    pip install numpy
    pip install Pillow
    pip install opencv-python
    pip install torchvision
    pip install colorthief
    pip install matplotlib

## All necessary Files and Folders:
'run.py'; 'model.py'; 'resnet.py'; 

'model' Folder; 'cascade' Folder; 'files' Folder; 'files_temp' Folder

##  pycache:
'__pycache__' Folder will be created by executing the 'run.py'


###  Inspired by:
[GithubRealFan](https://github.com/GithubRealFan)'s project: "[HairColorChange](https://github.com/GithubRealFan/HairColorChange)"
