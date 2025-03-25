import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:/Users/2273581/Downloads/Energy_dataset.csv')

# Show the shape of the dataset
shape = df.shape
print(f"Shape of the dataset: {shape}")

# Describe statistical summaries
statistical_summary = df.describe()
print("\nStatistical Summary:\n", statistical_summary)

# Check skewness
skewness = df.skew(numeric_only=True)
print("\nSkewness:\n", skewness)

# Detect outliers using a box plot
plt.figure(figsize=(15, 10))
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
plt.xticks(rotation=90)
plt.title('Box Plot for Outlier Detection')
plt.show()