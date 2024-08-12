# Housing Price Prediction

## Project Overview

This project focuses on predicting housing prices using linear regression. The dataset includes various features such as the number of bedrooms, neighborhood, and other relevant information. The main goal is to build a predictive model that can estimate the price of a house based on these features.

## Dataset

The dataset used for this project is a housing price dataset that includes the following features:
- Neighborhood: The neighborhood where the house is located.
- Bedrooms: The number of bedrooms in the house.
- Price: The price of the house (target variable).
- Other Numerical Features: Various other features that contribute to the house's price.

## Steps Performed

1. Data Loading:
   - The dataset is loaded using `pandas` from a CSV file.

2. Data Exploration:
   - Displayed the first few rows of the dataset using `head()`.
   - Obtained dataset information and summary statistics using `info()` and `describe()`.

3. Data Preprocessing:
   - Checked and dropped any missing values in the dataset.
   - Displayed unique values in categorical features like `Neighborhood` and `Bedrooms`.
   - Removed non-numeric columns to focus on features relevant for linear regression.
   - Stripped extra spaces from column names to ensure consistency.
   - Dropped the `Neighborhood` column since it is categorical and not directly usable in regression.

4. Data Visualization:
   - House Price Distribution: Visualized the distribution of house prices using a histogram with a Kernel Density Estimate (KDE) overlay.
   - Correlation Matrix: Created a heatmap to visualize correlations between numerical features.

5. Data Splitting:
   - Split the dataset into features (`X`) and target variable (`y`).
   - Further split the data into training (80%) and testing (20%) sets to evaluate the model's performance.

6. Feature Scaling:
   - Standardized the features using `StandardScaler` to ensure that all features contribute equally to the model.

7. Model Building and Training:
   - Built a linear regression model using `LinearRegression` from scikit-learn.
   - Trained the model using the training data.

8. Model Evaluation:
   - Predicted housing prices on both the training and testing datasets.
   - Calculated performance metrics including Root Mean Squared Error (RMSE) and R-squared (R²) to evaluate the model's accuracy.

9. Visualization:
   - Visualized the actual vs predicted prices using a scatter plot, with a reference line to assess how close the predictions are to the actual values.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Seaborn
- Matplotlib
- scikit-learn

## How to Run

1. Ensure that the required libraries are installed.
2. Download or clone the repository.
3. Place the dataset (`housing_price_dataset.csv`) in the appropriate directory.
4. Run the script to perform the analysis, build the model, and generate visualizations.

## Conclusion

This project demonstrates the process of building a simple linear regression model to predict housing prices. The model's performance is evaluated using RMSE and R-squared metrics, providing insights into its accuracy and effectiveness.
