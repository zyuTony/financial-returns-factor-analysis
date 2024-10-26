import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats

# Sample data provided in the image
data = pd.read_csv('./data2.csv')
# Select only rows with sector = TECHNOLOGY
data = data[data['sector'] == 'MANUFACTURING']

# Features (X) and target (y)
X = data[['ocf_to_debt_ratio', 'fcf_yield', 'ev_fcf_ratio', 'r12m_pe_ratio', 'pb_ratio', 
          'r12m_ev_ebitda_ratio', 'peg_ratio', 'eps_yoy_growth', 'revenue_yoy_growth', 
          'gross_profit_yoy_growth', 'fcf_yoy_growth', 'fcf_yield_yoy_growth', 'real_gdp', 
          'spy_gdp_ratio', 'cpi_yoy', 'durables_yoy', 'nonfarm_payroll_yoy', 'real_gdp_yoy', 
          'retail_sales_yoy', 'unemployment_yoy']].fillna(0).astype(float)

y = data['f90d_return'].fillna(0).astype(float)

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
n = X_train.shape[0]
p = X_train.shape[1]
dof = n - p - 1
X_train_with_const = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
mse = np.sum((y_train - model.predict(X_train))**2) / dof
var_b = mse * (np.linalg.inv(np.dot(X_train_with_const.T, X_train_with_const)).diagonal())
sd_b = np.sqrt(var_b)
t_stat = model.coef_ / sd_b[1:]
p_values = [2 * (1 - stats.t.cdf(np.abs(t), dof)) for t in t_stat]

feature_impact = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'P-value': p_values
})
feature_impact['Significant'] = feature_impact['P-value'] < 0.05
feature_impact = feature_impact.sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Impact and Significance:")
print(feature_impact)

# # Visualize feature impact and significance
# plt.figure(figsize=(12, 6))
# bars = plt.bar(feature_impact['Feature'], feature_impact['Coefficient'])
# plt.title('Feature Impact on 30-day Forward Return')
# plt.xlabel('Features')
# plt.ylabel('Coefficient (Impact)')
# plt.xticks(rotation=45)

# # Color bars based on significance
# for i, bar in enumerate(bars):
#     if feature_impact.iloc[i]['Significant']:
#         bar.set_color('blue')
#     else:
#         bar.set_color('lightblue')

# plt.legend(['Significant', 'Not Significant'], loc='best')
# plt.tight_layout()
# plt.show()