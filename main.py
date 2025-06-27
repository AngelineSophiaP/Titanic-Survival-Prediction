import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
print(df.head())

df = df.dropna(subset = ['age', 'embarked'])
df['sex'] = df['sex'].map({'male' : 1 , 'female' : 0})
df['embarked'] = df['embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2})

df = df.drop(['who', 'adult_male', 'deck', 'embark_town', 'alive', 'class', 'alone'], axis=1)
print(df.head())

X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f" Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=["Not Survived", "Survived"], filled=True)
plt.show()
