## Importing necessary libraries
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# importing the titanic data set 
titanic_data= pd.read_csv('D:\Cod soft projects\Titanic-Dataset.csv')
titanic_data.head()
# previewing the last five rows of the data
titanic_data.tail()
# getting the shape of the data
print(f" This data has {titanic_data.shape[0]} rows and {titanic_data.shape[1]} columns")

#checking types of columns. 
titanic_data.info()
# getting the summary statistics
titanic_data.describe().T
# handling missing data 
titanic_data = titanic_data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace=True)
data=titanic_data
data.describe ().T
data.tail()

# checking for duplicates
duplicates = data.duplicated()

# Display rows with duplicates
print("Duplicate Rows except first occurrence:")
print(data[duplicates])

# Count the number of duplicates
num_duplicates = duplicates.sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")

# Convert categorical features to numerical
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Split the data into features (X) and target variable (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Assuming 'new_passenger_data' is a DataFrame containing features of a new passenger
# Make sure the features match the columns used during training

# Handle missing values if any
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

# Convert categorical features to numerical
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Make prediction for the new passenger
new_passenger_data = data.tail(1).drop('Survived', axis=1)  # Assuming the new passenger data is the last row in the DataFrame
new_passenger_prediction = model.predict(new_passenger_data)

# Display the prediction for the new passenger
if new_passenger_prediction[0] == 1:
    print("The new passenger is predicted to have survived.")
else:
    print("The new passenger is predicted to have not survived.")