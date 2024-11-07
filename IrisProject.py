import pandas as pd
import seaborn as sns
# matplotlib for visualization
import matplotlib.pyplot as plt
# Load Iris dataset
iris = sns.load_dataset('iris')
#print(iris.head())
#print(iris.info())
#print(iris.describe())
# Visualising data
sns.pairplot(iris, hue="species")
plt.show()
#print(iris.groupby('species').mean())