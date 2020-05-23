# Week 2
Install matplotlib, numpy and csv on your systems using 
```
pip install matplotlib, numpy, csv
```
## Linear regression with one variable
Once you've installed, give yourself a moment and go through the test data 'ex1data.txt'.
We'll be using the above file for the first exercise, which is implementing linear regression with one variable.
The main components of the first part are given as below: 
* **singleVariableMain.​py**
The main file for the program, which will be using the other files to run the program. Here, we load in the data, visualize it, and preprocess it for training.
* **computeCost.​py**
The main file for computing the cost between predicted and actual values.
* **gradientDescent.​py**
The main file for actually implementing the gradient descent algorithm. 
## Linear regression with two variables

We'll be using the data in 'ex1data2.txt' for this exercise.
* **multiVariableMain.​py**
The main file for this exercise which will use the other files to run the program. Here, multiple features are read in and accoomodated for by the program. 
* **featureNormalization.​py**
File for normalizing features by subtracting the values by the mean and dividing them by the standard deviation.
