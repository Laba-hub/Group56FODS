import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import scipy.stats as st

# generate datasets
rnd.seed(1)

#variable 1
var1 = 20* rnd.randn(1000) + 100.

#variable 2
var2 = var1 + (10* rnd.randn(1000) + 50)

print("Mean of var1: ", np.mean(var1))

print("mean of var 2:", np.mean(var2))

#plot scatterplot
plt.scatter(var1, var2)
plt.show()

#calculate the correlationn coefficiennt between var1 and var2
corr, _ = st.pearsonr(var1, var2) #, _ place holder
print("The correlation coefficient between var1 and var2: ",corr)

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#read .csv file from the website
data = pd.read_csv('https://raw.githubusercontent.com/drazenz/heatmap/master/autos.clean.csv')

print(data.info())
# calculate pairwise correlation correlation between all variables
result = data.corr()

#visualize the correlation coefficient
ax = sns.heatmap(
    result,
    vmin = -1,
    vmax = 1,
    center = 0,
    cmap = sns.diverging_palette(20, 220, n=200),
    square = True
)

#further customize the figure object
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45, 
    horizontalalignment = 'right' 
)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

mydata = sns.load_dataset("iris")
ax = sns.boxplot(data=mydata,
                 orient="h",
                 palette="set2"
                 whis=1.5)
plt.show()
