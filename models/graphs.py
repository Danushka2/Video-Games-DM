# -*- coding: utf-8 -*-

# Importing
import pandas as pd
import seaborn as sns
import numpy as np
import os
from dataset.cleanDs import cleanDs
import matplotlib.pyplot as plt

#assign cleaned database here and use it


################## get data from cleanDs    ############################
cleanDataset = cleanDs()
cleanedDS = cleanDataset.clean_db()


# Declaring the points for first line plot
X1 = cleanedDS['Genre']
Y1 = cleanedDS['Global_Sales']

# Setting the figure size
fig = plt.figure(figsize=(10,5))

# plotting the first plot
plt.bar(X1, Y1, align='center', alpha=0.5)

# Labeling the X-axis 
plt.xlabel('Genre') 

# Labeling the Y-axis 
plt.ylabel('Global_Sales') 

# Give a title to the graph
plt.title('Total sales of the Genres') 

# Show a legend on the plot 
plt.legend() 

#Saving the plot as an image
fig.savefig('../static/images/line_plot.jpg', bbox_inches='tight', dpi=150)

