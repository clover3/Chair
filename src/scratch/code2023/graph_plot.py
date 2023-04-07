

import matplotlib.pyplot as plt
import numpy as np

b = 2 # set the value of b
x = np.linspace(0, 10, 1000) # generate 1000 points between 0 and 10

y = (x+0.1*x)/(x+0.1 * (1-2 + 2*1)) # calculate the corresponding values of y

plt.plot(x, y) # plot the function
plt.xlabel('x') # add a label to the x-axis
plt.ylabel('y') # add a label to the y-axis
plt.title('Graph of y = (x+0.1+x)/(x+0.1 * (1-2 + 2*1))') # add a title to the graph
plt.show() # display the graph
