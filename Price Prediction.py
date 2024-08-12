# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load Dataset
file_path = r"C:\Users\Teddy\OneDrive\Documents\Predictive Modeling\housing_price_dataset.csv"
data = pd.read_csv(file_path)

# Data Exploration Steps
# Display the first 5 rows of the dataset
print(data.head())

# Display dataset information
print(data.info())

# Summary statistics of numerical features
print(data.describe()) #provides summary statistics.

# Data Preprocessing Steps
# Check for missing values
print(data.isnull().sum())
data=data.dropna()#drop missing value(if any)

# Check unique values in categorical features
print(data['Neighborhood'].value_counts())
print(data['Bedrooms'].value_counts())
 
# Identifying Duplicate
print(data.duplicated())

# Data visualization
plt.figure(figsize=(10,6))
sns.histplot(data['Price'],kde=True)#Kernel Density Estimate (KDE) plot over the histogram
plt.title('Distribution of House Prices')
plt.xlabel('price')
plt.ylabel('frequency')
plt.show()

#Dropping non numeric columns
non_numeric_data = data.select_dtypes(exclude=[float,int])
print(non_numeric_data)
data.columns = data.columns.str.strip()#stripping extra space
data=data.drop('Neighborhood',axis=1)

#Heat Map
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')#annotation,colormap
plt.title('Correlation Matrix')
plt.show()

#splitting data into features and target varibale
x = data.drop('Price',axis=1) # X contains all columns except 'price'
y = data['Price'] # y contains only the 'price' column
#splitting the dataset into testing(20%) and training (80%)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#purpose:StandardScaler standardizes features by removing the mean and scaling to unit variance. This is also known as Z-score normalization.
#Feature Scaling
Scaler=StandardScaler()#creating instance
x_train_scaled=Scaler.fit_transform(x_train)#fit_transform calculates the necessary statistics
x_test_scaled=Scaler.transform(x_test)
print(x)
print(x_test_scaled)



#model building and training
model = LinearRegression()#This creates an instance of the LinearRegression class. Linear Regression is a fundamental algorithm used for predicting a continuous target variable based on one or more features.
model.fit(x_train_scaled,y_train)

#model evaluation 
y_train_prediction = model.predict(x_train_scaled)
y_test_prediction = model.predict(x_test_scaled)
print(y_train.values)
print(y_train_prediction)

#calculate performance metrices
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_prediction))#RMSE is a measure of how well the model's predictions match the actual observed data.
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_prediction))
train_r2 = r2_score(y_train, y_train_prediction) #R² is a statistical measure that represents the proportion of the variance for the dependent variable (Y) that's explained by the independent variables (X) in the model.
test_r2 = r2_score(y_test, y_test_prediction)

print(train_rmse)
print(test_rmse)
print(train_r2)
print(test_r2)

#visualizing actual vs predicted price
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test_prediction, alpha=0.2)# alpha= transparency level of the points in the scatter plot.
plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()], '--r', linewidth = 2 )
plt.xlabel("Actual")
plt.ylabel("predicted")
plt.title("Actual vs Predicted")
plt.show()