import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest



#dataset with 2 cols and 15 rows
df = pd.DataFrame({'StudentId':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
                  'Marks': [67, 56, 66, 74, 70, 45, 55, 59, 69, 99, 68, 51, 60, 69, 50]})

print(df.head())
print(df.shape)
print(df.info())


#let’s use the boxplot to see the outlier
sns.boxplot(df.Marks)
plt.show()

#create an Isolation Forest Model to detect the outlier in the dataset.

isolation_forest = IsolationForest(contamination=0.01)  # Set contamination to the expected proportion of outliers
isolation_forest.fit(df[['Marks']])


# Predict outliers
outlier_preds = isolation_forest.predict(df[['Marks']])

# Add outlier predictions to DataFrame
df['Outlier'] = outlier_preds
# Print DataFrame with outlier predictions
print(df)