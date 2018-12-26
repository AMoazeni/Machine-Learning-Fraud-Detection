# Introduction

Take me to the [Fraud Detection Code](https://github.com/AMoazeni/Machine-Learning-Fraud-Detection/blob/master/Code/Fraud%20Detection.py) for Fraud Detection!


<br></br>
This article explores a method that detects fraudulent credit card activity in a bank dataset. We are going to use an Unsupervised Machine Learning technique called Self Organizing Map (SOM) to achieve this. The SOM generates a fraud probability for each Credit Card activity, which can be forwarded to the fraud investigation department. The fraud probability table can also be used to alert the customer instantly through text or email if they have previously set up Multi Factor Authentication (MFA).


<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Fraud-Detection/master/Jupyter%20Notebook/Images/01%20-%20Credit%20Card.gif" width=50% alt="Credit-Card"></div>



<br></br>

# Self Organizing Map (SOM)

Self Organizing Maps are used to reveal correlations that are not easily identifiable, by decreasing dimensionality in the dataset (condense data from several columns into a few columns). It was popularized by Teuvo Kohonen who is a Finnish researcher. Unlike supervised learning, unsupervised SOMs don't have activation functions, labeled datasets, and don't require back propagation. They find patterns in your data and group similar data points together.


<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Fraud-Detection/master/Jupyter%20Notebook/Images/02%20-%20SOM.png" width=75% alt="Self-Organizing-Map"></div>


<br></br>
SOMs retain the topology input data by masking a map onto the dataset. Put simply, SOMs find patterns in data by calculating and minimizing the "distance" between SOM map points and data points. They repeat this calculation until the data is fully mapped, the following figure demonstrates this.


<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Fraud-Detection/master/Jupyter%20Notebook/Images/03%20-%20SOM%20Map.png" width=75% alt="SOM-Map"></div>



<br></br>

# SOM Algorithm

### STEP 1
Start with a dataset consisting of 'n' features (independent variables).

### STEP 2
Create a grid composed of nodes, each one having a weight vector of 'n' features elements.

### STEP 3
Randomly initialize the values of the weight vectors to small numbers close to O (but not equal to 0).

### STEP 4
Select a random observation point from the dataset.

### STEP 5
Compute the Euclidean distances from this point to the different neurons in the network.

### STEP 6
Select the neuron that has the minimum distance to the point. This neuron is called the winning node.

### STEP 7
Update the weights of the winning node to move it closer to the starting point.

### STEP 8
Use a Gaussian neighborhood function of means for the winning node, and update the weights of the winning node neighbors to move them closer to the point. The neighborhood radius is the sigma in your Gaussian function.

### STEP 9
Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning) or after a batch of observations (Batch Learning). Repeat until the network converges to a point and the neighborhood stops decreasing.



<br></br>

# Results

Run the first part of the code to generate the following Self Organizing Map result. Assuming that most applications are truthful (majority dark cells), the outliers must be fraudulent (white cells). Plug in the coordinates of the white cells into the 'fraud' variable. The results will be saved into the Data folder called 'Fraud_Results.csv'.


<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Fraud-Detection/master/Jupyter%20Notebook/Images/04%20-%20Results.png" width=75% alt="SOM-Results"></div>


<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Fraud-Detection/master/Jupyter%20Notebook/Images/05%20-%20SOM%20Frauds.png" width=75% alt="SOM-Frauds"></div>




<br></br>

# Code

1. Install [Anaconda](https://www.anaconda.com/download/).
2. Download this repository and navigate to it.
3. Include the [minisom library](https://github.com/JustGlowing/minisom) in your working directory.
4. Click 'Run' to step through the [Fraud Detection Jupyter Notebook](https://github.com/AMoazeni/Machine-Learning-Fraud-Detection/blob/master/Jupyter%20Notebook/Fraud%20Detection.ipynb) code.


<br></br>
```shell
$ git clone https://github.com/AMoazeni/Machine-Learning-Fraud-Detection.git
$ cd Machine-Learning-Fraud-Detection
```



<br></br>

# Happy Coding!

Check out [AMoazeni's Github](https://github.com/AMoazeni/) for more Machine Learning, and Robotics repositories.

<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Word-Count/master/Jupyter%20Notebook/Images/06%20-%20Cat%20Typing.gif" width=40% alt="Cat-Typing"></div>

