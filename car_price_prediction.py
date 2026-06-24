# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
data = pd.read_csv("car data.csv")

# Show First 5 Rows
print("First 5 Rows:\n")
print(data.head())

# Dataset Information
print("\nDataset Info:\n")
print(data.info())

# Check Missing Values
print("\nMissing Values:\n")
print(data.isnull().sum())

# Convert Categorical Data into Numbers
data = pd.get_dummies(data, drop_first=True)

# Input Features
X = data.drop(['Selling_Price'], axis=1)

# Output Target
y = data['Selling_Price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict Prices
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

print("\nR2 Score:", score)

# Error
mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", mae)

# Actual vs Predicted Graph
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Car Prices")

plt.show()

# Heatmap
plt.figure(figsize=(10,6))

sns.heatmap(data.corr(), cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

sample = X_test.iloc[0:1]

prediction = model.predict(sample)

print("\nPredicted Price:", prediction)

sns.countplot(x='Fuel_Type_Petrol', data=data)

plt.title("Fuel Type Distribution")

plt.show()

sns.pairplot(data)

plt.show()