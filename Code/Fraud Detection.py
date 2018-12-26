# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('../Data/Credit_Card_Applications.csv')
# All columns except the last are attributes (not dependent value)
X = dataset.iloc[:, :-1].values
# People whos applications were approved (1) or not approved (0)
y = dataset.iloc[:, -1].values


# Feature Scaling (Normalization)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
# 'x=10' 'y=10' defines a 10x10 SOM grid, too small and you miss patterns
# 'input_len = 15' is the number of columns to look at in X
# 'sigma' is the radius of the different radius in the grid, default value = 1
# 'learning_rate' hyprer-parameter which decided how much the rates are updated. Higher is faster
# 'learning_rate' and 'decay_function'deerming rate of convergence, default value = 0.5
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# Initialize weights to small numbers close to (but not equal to) zero
som.random_weights_init(X)
# Train SOM with a 'num_iteration' resembling epochs
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()



# Finding the frauds
# Save the results into a dictionary
mappings = som.win_map(X)

# Select fraudulent outlier coordinates (white boxes)
frauds = np.concatenate((mappings[(6,2)], mappings[(6,3)]), axis = 0)

# Perform inverse scaling to retrieve original values
frauds = sc.inverse_transform(frauds)

# Save frauds to file and display the results
frauds = pd.DataFrame(frauds)
frauds.to_csv('../Data/Fraud_Results.csv')
display(frauds)

