# Week 4
Install OpenCV on your system using the following method:
```
pip3 install opencv-python
```
This assignment was created using openCV version : 4.2.0 (I'm proud)
## Support Vector Machines

This week we'll be using Support Vector machines to create a mini project of sorts. We'll be working on improving this project later on as well.
There are total four components in the file

* **Main.​py**
 The main file for the program, here we will be loading in the SVM from sklearn and then train and store the model in a different file, which can be accessed by other files.
 
 * **Detector.​py**
 The file will load in our image to test it on, and the classifier model, and predict the values.
 
 * **photo8.​jpg**
 The sample File I'll be using to test the work.
 
 * **digits_cls.​pkl** 
 The file you'll be generating by running Main.​py to save the classifier, and the file ​Detector.​py will  read to load in the classifier. This file 'stores' the model.

## Optional Exercise
Right now you're loading in data from one file, if you've gone through the openCV documentation,try and take the input from a livestream input via the webcam
