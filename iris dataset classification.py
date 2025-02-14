# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import zscore

# Loading the Dataset
data = load_iris()
# Converting the data into a DataFrame
iris_df = pd.DataFrame(
    data=np.c_[data['data'], data['target']],
    columns=data['feature_names'] + ['target']
)

# Exploration of the Data
print("Checking for missing values:")
print(iris_df.isnull().sum())

# Information about the dataset
print("Dataset Info:")
iris_df.info()

# Summary of the dataset
print("Summary Statistics:")
print(iris_df.describe())

# Data Cleaning Process
iris_df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '') for col in iris_df.columns]
print("Updated Column Names:")
print(iris_df.columns)

# Calculating central tendencies for all features
print("Mean Values:")
print(iris_df.mean())
print("Median Values:")
print(iris_df.median())
print("Standard Deviation:")
print(iris_df.std())

# Visualizing the Data
print("Histograms!")
iris_df.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()

# Pairplot
print("Pairplot!")
sns.pairplot(iris_df, hue='target', diag_kind='kde')
plt.show()

# Boxplots
print("Boxplots!")
plt.figure(figsize=(10, 5))
sns.boxplot(data=iris_df.iloc[:, :-1])
plt.title("Feature Value Ranges")
plt.show()

# Scatter plot
print("Scatter Plot!")
plt.scatter(iris_df['sepal_length_cm'], iris_df['petal_length_cm'], c=iris_df['target'], cmap='viridis')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Target")
plt.show()

# Correlation heatmap
print("Correlation Heatmap!")
correlation_matrix = iris_df.iloc[:, :-1].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlations")
plt.show()

# Checking outliers using z-scores
z_scores = np.abs(zscore(iris_df.iloc[:, :-1]))
outliers = (z_scores > 3).sum()
print("Outliers in each feature:")
print(outliers)