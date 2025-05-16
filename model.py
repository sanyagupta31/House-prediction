import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv(r'house-prices-advanced-regression-techniques/train.csv')

# Drop rows with missing values in selected columns
df = df[['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'SalePrice']].dropna()

# Feature matrix and target
X = df[['GrLivArea', 'TotalBsmtSF', 'OverallQual']]
y = df['SalePrice']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model trained and saved as model.pkl")
