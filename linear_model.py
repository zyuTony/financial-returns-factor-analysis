import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Sample data provided in the image
data = pd.read_csv('./data2.csv')
sectors = [
    # 'ENERGY & TRANSPORTATION',
    'REAL ESTATE & CONSTRUCTION',
    # 'TRADE & SERVICES',
    # 'LIFE SCIENCES',
    # 'MANUFACTURING',
    # 'FINANCE',
    # 'TECHNOLOGY'
]

# Filter data to include only the specified sectors
data = data[data['sector'].isin(sectors)]

# To use all sectors, keep the line above commented out
# Features (X) and target (y)
X = data[['ocf_to_debt_ratio', 'fcf_yield', 'ev_fcf_ratio', 'r12m_pe_ratio', 'pb_ratio',
                 'r12m_ev_ebitda_ratio', 'peg_ratio', 'eps_yoy_growth', 'revenue_yoy_growth',
                 'gross_profit_yoy_growth', 'fcf_yoy_growth', 'fcf_yield_yoy_growth', 'spy_gdp_ratio',
                 'cpi_yoy', 'durables_yoy', 'nonfarm_payroll_yoy', 'real_gdp_yoy', 'retail_sales_yoy',
                 'unemployment_yoy', 'r10d_return', 'r30d_return', 'r90d_return', 'r180d_return',
                 'r365d_return']].fillna(0).astype(float)
y = data['f180d_return'].fillna(0).astype(float)


# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
# Analyze feature importance, directional impact, and statistical significance
X_with_const = sm.add_constant(X_scaled)
model = sm.OLS(y, X_with_const).fit()

# Create a summary of the model results
summary = model.summary()
print(summary)

# Extract feature importance and significance
feature_impact = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.params[1:],
    'Std Error': model.bse[1:],
    't-statistic': model.tvalues[1:],
    'P-value': model.pvalues[1:]
})

feature_impact['Significant'] = feature_impact['P-value'] < 0.05
feature_impact = feature_impact.sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Impact and Significance:")
print(feature_impact)

# Visualize feature importance
plt.figure(figsize=(12, 8))
feature_impact.plot(x='Feature', y='Coefficient', kind='bar', ax=plt.gca())
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Perform predictions on test set
X_test_const = sm.add_constant(X_test)
y_pred = model.predict(X_test_const)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Residual analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# Q-Q plot for normality check
fig, ax = plt.subplots(figsize=(10, 6))
sm.qqplot(residuals, ax=ax, line='45')
ax.set_title('Q-Q Plot of Residuals')
plt.tight_layout()
plt.show()

# Durbin-Watson test for autocorrelation
dw_statistic = sm.stats.durbin_watson(model.resid)
print(f"\nDurbin-Watson statistic: {dw_statistic:.4f}")

# Variance Inflation Factor (VIF) for multicollinearity
vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print("\nVariance Inflation Factors:")
print(vif)