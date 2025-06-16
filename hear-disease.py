import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Setup ---

# Change this if your file is in a different folder:
file_path = 'heart.csv'

# Check if file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"CSV file not found at {file_path}")

# Load data
data = pd.read_csv(file_path)

# Show first 5 rows and data shape
print("First 5 rows of the dataset:")
print(data.head())
print(f"\nDataset shape: {data.shape}")

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Check distribution of target variable
print("\nTarget variable value counts:")
print(data['target'].value_counts())


# --- Feature Engineering ---

# Add a new feature: cholesterol to age ratio
data['chol_age_ratio'] = data['chol'] / data['age']

# Prepare features and labels
X = data.drop(columns='target', axis=1)
Y = data['target']
print("\nFeature sample:")
print(X.head())
print("\nTarget sample:")
print(Y.head())


# --- Split data into training and test sets ---

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=3
)
print(f"\nFull dataset shape: {X.shape}")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# --- Model training ---

model = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence
# Alternative: use RandomForestClassifier()
# model = RandomForestClassifier(random_state=3)

model.fit(X_train, Y_train)


# --- Evaluation on training data ---

train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"\nTraining accuracy: {train_accuracy:.4f}")


# --- Evaluation on test data ---

test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nClassification Report on Test Data:")
print(classification_report(Y_test, test_predictions))

print("Confusion Matrix on Test Data:")
print(confusion_matrix(Y_test, test_predictions))


# --- Predict on new data ---

# Example input data
# Make sure features order is exactly as in X.columns
chol_age_ratio_new_data = 138 / 62

input_data = (
    63,   # age
    1,    # sex
    3,    # cp (chest pain type)
    145,  # trestbps (resting blood pressure)
    233,  # chol (serum cholesterol)
    1,    # fbs (fasting blood sugar)
    0,    # restecg (resting electrocardiographic results)
    150,  # thalach (maximum heart rate achieved)
    0,    # exang (exercise induced angina)
    2.3,  # oldpeak (ST depression induced by exercise relative to rest)
    0,    # slope (slope of the peak exercise ST segment)
    0,    # ca (number of major vessels colored by fluoroscopy)
    1,    # thal (thalassemia)
    chol_age_ratio_new_data  # chol_age_ratio (new feature)
)
# Convert input data to numpy array and reshape
input_data_array = np.asarray(input_data).reshape(1, -1)

# Predict
prediction = model.predict(input_data_array)

if prediction[0] == 0:
    print("\nPrediction: Person does NOT have heart disease :)")
else:
    print("\nPrediction: Person HAS heart disease :(")

    
# --- Visualization ---

plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='chol_age_ratio', hue='target', kde=True)
plt.title('Distribution of Cholesterol to Age Ratio by Heart Disease Status')
plt.xlabel('Cholesterol to Age Ratio')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='target', y='chol_age_ratio')
plt.title('Cholesterol to Age Ratio Distribution by Heart Disease Status')
plt.xlabel('Heart Disease (0: No, 1: Yes)')
plt.ylabel('Cholesterol to Age Ratio')
plt.show()