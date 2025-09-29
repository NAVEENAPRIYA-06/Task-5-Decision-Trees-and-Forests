# tree_models.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
import numpy as np # Needed for array handling

# Initialize variables to avoid NameError
X, y, feature_names, class_names, dataset_name = None, None, None, None, None

# --- 1. Load Data (Corrected) ---
print("--- Starting Data Loading and Preprocessing ---")

# --- TRY: Load External CSV ---
try:
    data = pd.read_csv('heart.csv')
    
    data = data.dropna()
    
    X = data.drop('target', axis=1)
    y = data['target']
    feature_names = X.columns.tolist()
    # Assuming two classes (0 and 1) for classification
    class_names = ['No Disease', 'Disease']
    dataset_name = "Heart Disease CSV"

# --- EXCEPT: Fallback to Built-in Iris Dataset ---
except FileNotFoundError:
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    feature_names = iris.feature_names
    class_names = [str(c) for c in iris.target_names]
    dataset_name = "Iris"
    print("Dataset 'heart_disease.csv' not found. Using the built-in Iris dataset.")

# --- 2. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Dataset used: {dataset_name}. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print("--- Data Split Complete ---")


# --- 3. Train Decision Tree (Full Depth - Likely Overfitted) ---
print("\n--- Training Decision Tree (Full Depth) ---")
dt_classifier_full = DecisionTreeClassifier(random_state=42)
dt_classifier_full.fit(X_train, y_train)

# Predict and evaluate
y_pred_dt_full = dt_classifier_full.predict(X_test)
accuracy_dt_full = accuracy_score(y_test, y_pred_dt_full)

print(f"Decision Tree Test Accuracy (Full Depth): {accuracy_dt_full:.4f}")

# Save visualization of the full tree
plt.figure(figsize=(18, 12))
plot_tree(
    dt_classifier_full,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title(f"Decision Tree Visualization (Full Depth, Accuracy: {accuracy_dt_full:.4f})")
plt.savefig('01_decision_tree_full.png')
# graphviz part removed for brevity, as plot_tree is often sufficient. If needed, uncomment/re-add.


# --- 4. Analyze Overfitting and Control Tree Depth (Objective 2) ---
# Train a limited depth tree to reduce overfitting (e.g., max_depth=3)
print("\n--- Training Decision Tree (Limited Depth: 3) ---")
dt_classifier_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier_limited.fit(X_train, y_train)

y_pred_dt_limited = dt_classifier_limited.predict(X_test)
accuracy_dt_limited = accuracy_score(y_test, y_pred_dt_limited)

print(f"Decision Tree Test Accuracy (Depth=3): {accuracy_dt_limited:.4f}")

# Save visualization of the limited tree
plt.figure(figsize=(15, 10))
plot_tree(
    dt_classifier_limited,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title(f"Decision Tree Visualization (Depth=3, Accuracy: {accuracy_dt_limited:.4f})")
plt.savefig('02_decision_tree_depth_3.png')
print("Saved visualization of limited tree to '02_decision_tree_depth_3.png'")

# --- Comparison for Overfitting Analysis ---
print("\n--- Overfitting Analysis ---")
train_acc_full = dt_classifier_full.score(X_train, y_train)
train_acc_limited = dt_classifier_limited.score(X_train, y_train)

print(f"Full Tree: Train Acc = {train_acc_full:.4f}, Test Acc = {accuracy_dt_full:.4f}")
print(f"Depth 3 Tree: Train Acc = {train_acc_limited:.4f}, Test Acc = {accuracy_dt_limited:.4f}")
print("A large gap between Train and Test accuracy in the Full Tree indicates **overfitting**.")