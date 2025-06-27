
# 🚢 Titanic Survival Prediction using Decision Tree

## 📌 Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning. We use the 
**Decision Tree Classifier**, a supervised learning algorithm that works like a flowchart to make decisions based on feature conditions.

The dataset includes passenger details like age, gender, ticket class, number of siblings/spouses, and more.
By analyzing these features, the model predicts the probability of survival.

---

## 🧠 Problem Type

**Binary Classification Problem**

* Target: `Survived` (1 = Survived, 0 = Did not survive)

---

## 🧰 Tools and Libraries Used

| Task             | Tool/Library                                                                         |
| ---------------- | ------------------------------------------------------------------------------------ |
| Data Handling    | `pandas`, `numpy`                                                                    |
| Visualization    | `seaborn`, `matplotlib`                                                              |
| Model Training   | `DecisionTreeClassifier` from `sklearn.tree`                                         |
| Model Evaluation | `accuracy_score`, `confusion_matrix`, `classification_report` from `sklearn.metrics` |

---

## 📊 Dataset

* Source: Titanic dataset from `seaborn`
* Features used (after preprocessing):

  * `pclass` (Ticket class)
  * `sex` (Gender)
  * `age`
  * `sibsp` (Siblings/Spouses aboard)
  * `parch` (Parents/Children aboard)
  * `fare` (Ticket Fare)
  * `embarked` (Port of Embarkation)

---

## 🧪 Steps Performed

1. **Data Loading**
   Loaded Titanic dataset using Seaborn.

2. **Data Cleaning & Preprocessing**

   * Removed rows with null values in key columns (`age`, `embarked`).
   * Encoded categorical features like `sex` and `embarked` into numerical format.

3. **Feature & Label Definition**

   * `X`: Selected feature columns
   * `y`: Target (`survived`)

4. **Train-Test Split**

   * Used `train_test_split()` with 80% training and 20% testing.

5. **Model Training**

   * Trained a Decision Tree Classifier on the training data.

6. **Prediction**

   * Predicted survival on the test dataset.

7. **Evaluation**

   * Accuracy score
   * Confusion matrix
   * Classification report (precision, recall, f1-score)

---

## 🎯 Results

* ✅ **Model Used**: `DecisionTreeClassifier()`
* 📈 **Accuracy**: *(e.g., 0.78 — fill in your result)*
* 🧾 **Classification Report**:

  * Precision, Recall, F1-score printed per class
* 🔍 **Confusion Matrix**: Displays true/false positives and negatives

---

## 📚 What I Learned

* How to preprocess real-world data for machine learning
* How a Decision Tree splits features to make predictions
* Difference between binary classification and regression problems
* How to interpret confusion matrix and classification metrics

---

## ✅ Conclusion

Decision Tree is a powerful and interpretable model. With the Titanic dataset, it provided good predictions and 
demonstrated how simple conditions can help make meaningful decisions in real life.

---
