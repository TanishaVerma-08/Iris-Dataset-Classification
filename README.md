# Iris Dataset Classification

## Overview
This project explores the **Iris dataset**, a well-known dataset in the field of machine learning and data analysis. The dataset consists of 150 samples with four features:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)
- **Target Class** (Setosa, Versicolor, Virginica)

The objective of this project is to **load, clean, analyze, and visualize** the dataset using Python libraries such as **pandas, NumPy, matplotlib, seaborn, and SciPy**.

## Features Implemented
### 1. **Dataset Exploration**
- Load the Iris dataset from `sklearn.datasets`.
- Convert it into a Pandas **DataFrame**.
- Check for **missing values** and dataset structure using `.info()`.
- Generate **summary statistics** using `.describe()`.

### 2. **Data Cleaning**
- Renamed columns to remove spaces and parentheses for better handling.
- Handled any potential missing values (though none exist in this dataset).

### 3. **Statistical Analysis**
- Computed **mean, median, and standard deviation** for all numerical features.
- Checked for **outliers** using **z-score analysis** (values beyond 3 standard deviations).

### 4. **Data Visualization**
- **Histograms**: Show feature distributions.
- **Pairplot**: Display relationships between features grouped by target classes.
- **Boxplots**: Identify outliers and feature ranges.
- **Scatter Plot**: Analyze relationships between Sepal Length and Petal Length.
- **Correlation Heatmap**: Show feature correlations.

## Requirements
Ensure you have the following libraries installed before running the script:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/TanishaVerma-08/Iris-Dataset-Classification/tree/main
   ```
2. Navigate to the project directory:
   ```bash
   cd iris-dataset-analysis
   ```
3. Run the script:
   ```bash
   python "iris dataset classification.py"
   ```

## Example Output
The script generates:
- Summary statistics of the dataset.
- Various plots to analyze feature distributions and relationships.
- A correlation heatmap.
- Outlier detection results based on z-score.

## License
This project is open-source and available under the **MIT License**.

## Author
Tanisha Verma

---
Feel free to reach out for any questions or contributions!

