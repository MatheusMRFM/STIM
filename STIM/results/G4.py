import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.linear_model import LinearRegression # pip install scikit-learn

# Constants
PLT_DPI = 300
FONT_SIZE = 14
POINT_SIZE = 10

# Plot Size (FONT_SIZE OR: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large')
params = {'legend.fontsize': FONT_SIZE,
          'axes.labelsize':  FONT_SIZE,
          'axes.titlesize':  FONT_SIZE,
          'xtick.labelsize': FONT_SIZE,
          'ytick.labelsize': FONT_SIZE}
pylab.rcParams.update(params)

# Read Data Frame
df = pd.read_csv('time_large.dat', sep='\t')

# Create Numpy Arrays
x = df.iloc[:, 0].values.reshape(-1, 1) # Reshape: -1 means it will predict the dimentions of the rows and 1 means that the array will have just one column
y = df.iloc[:, 1].values.reshape(-1, 1) # Reshape: -1 means it will predict the dimentions			 of the rows and 1 means that the array will have just one column

# Perform Regression
linear_regression = LinearRegression()
linear_regression.fit(x, y)
y_prediction = linear_regression.predict(x)

# Results
fig, axes = plt.subplots(dpi=PLT_DPI)
plt.xlabel("Number of Edges")
plt.ylabel("Time (s)")
plt.scatter(x, y, color='tab:blue', s=POINT_SIZE)
plt.plot(x, y_prediction, color='tab:red')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('regression.png')
plt.close()
