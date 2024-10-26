import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Load the updated CSV data into a DataFrame
df = pd.read_csv('./data.csv')

# Drop date and symbol columns
df = df.drop(['month', 'symbol'], axis=1)

# Convert numeric columns to float, replacing commas with periods
numeric_columns = ['r1m_return', 'r3m_return', 'r6m_return', 'r9m_return', 'r12m_return', 'f1m_return']
for col in numeric_columns:
    df[col] = df[col].str.replace(',', '.').astype(float)

# Define all independent variables and target variable
all_vars = ['r1m_return', 'r3m_return', 'r6m_return', 'r9m_return', 'r12m_return']
target_var = 'f1m_return'

# Prepare the data
X = df[all_vars]
y = df[target_var]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model with all features
X_all = sm.add_constant(X)
final_model = sm.OLS(y, X_all).fit()

# Print the results
print(f"\nModel for predicting {target_var} using all variables:")
print(f"Variables used: {', '.join(all_vars)}")
print("\nRegression results:")
print(final_model.summary())

# Calculate and print AIC and BIC
print(f"\nAIC: {final_model.aic:.2f}")
print(f"BIC: {final_model.bic:.2f}")

# Calculate and print correlations for all variables
print("\nCorrelations with future 1-month return:")
for x_col in all_vars:
    correlation = df[x_col].corr(df[target_var])
    print(f"Correlation between {x_col} and {target_var}: {correlation:.4f}")

print("\nDescriptive Statistics:")
print(df[all_vars + [target_var]].describe())

# Evaluate the model on the test set
X_test_all = sm.add_constant(X_test)
y_pred = final_model.predict(X_test_all)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error on test set: {mse:.4f}")
