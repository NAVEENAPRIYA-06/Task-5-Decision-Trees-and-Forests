# tree_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import graphviz # Imported but not explicitly used for rendering to avoid dependency issues

# Initialize variables to avoid NameError if file is missing
X, y, feature_names, class_names, dataset_name = None, None, None, None, None

# --- 1. Load Data and Preprocessing ---
print("--- Starting Data Loading and Preprocessing ---")

# --- TRY: Load External CSV (Assumes 'heart_disease.csv' is present) ---
try:
    data = pd.read_csv('heart_disease.csv')
    
    # Simple preprocessing for Heart Disease UCI (assuming 'target' is the class column)
    data = data.dropna()
    
    X = data.drop('target', axis=1)
    y = data['target']
    feature_names = X.columns.tolist()
    class_names = ['No Disease', 'Disease']
    dataset_name = "Heart Disease CSV"
    print(f"Successfully loaded external dataset: {dataset_name}.")

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

# Ensure X and y were successfully assigned before proceeding
if X is None or y is None:
    raise RuntimeError("Data loading failed. Check your CSV file or the Iris import.")

# --- 2. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Dataset used: {dataset_name}. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print("--- Data Split Complete ---")


# ----------------------------------------------------------------------
# --- 3. Train Decision Tree (Full Depth - Objective 1 & Overfitting Base) ---
# ----------------------------------------------------------------------
print("\n--- Training Decision Tree (Full Depth) ---")
dt_classifier_full = DecisionTreeClassifier(random_state=42)
dt_classifier_full.fit(X_train, y_train)

# Evaluation
y_pred_dt_full = dt_classifier_full.predict(X_test)
accuracy_dt_full = accuracy_score(y_test, y_pred_dt_full)
train_acc_full = dt_classifier_full.score(X_train, y_train)

print(f"Decision Tree Train Accuracy (Full): {train_acc_full:.4f}")
print(f"Decision Tree Test Accuracy (Full): {accuracy_dt_full:.4f}")

# Visualization (Saves to file)
plt.figure(figsize=(18, 12))
plot_tree(dt_classifier_full, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=8)
plt.title(f"Decision Tree Visualization (Full Depth, Test Acc: {accuracy_dt_full:.4f})")
plt.savefig('01_decision_tree_full.png')


# ----------------------------------------------------------------------
# --- 4. Analyze Overfitting and Control Tree Depth (Objective 2) ---
# ----------------------------------------------------------------------
print("\n--- Training Decision Tree (Limited Depth: 3) ---")
dt_classifier_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier_limited.fit(X_train, y_train)

# Evaluation
y_pred_dt_limited = dt_classifier_limited.predict(X_test)
accuracy_dt_limited = accuracy_score(y_test, y_pred_dt_limited)
train_acc_limited = dt_classifier_limited.score(X_train, y_train)

print(f"Decision Tree Train Accuracy (Depth=3): {train_acc_limited:.4f}")
print(f"Decision Tree Test Accuracy (Depth=3): {accuracy_dt_limited:.4f}")

# Overfitting comparison
print("\n--- Overfitting Analysis ---")
print(f"Full Tree Gap (Train - Test): {train_acc_full - accuracy_dt_full:.4f}")
print(f"Depth 3 Tree Gap (Train - Test): {train_acc_limited - accuracy_dt_limited:.4f}")

# Visualization (Saves to file)
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier_limited, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10)
plt.title(f"Decision Tree Visualization (Depth=3, Test Acc: {accuracy_dt_limited:.4f})")
plt.savefig('02_decision_tree_depth_3.png')


# ----------------------------------------------------------------------
# --- 5. Train Random Forest & Compare Accuracy (Objective 3) ---
# ----------------------------------------------------------------------
print("\n--- Training Random Forest Classifier ---")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluation
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Test Accuracy: {accuracy_rf:.4f}")

# Comparison
print("\n--- Model Comparison (Test Accuracy) ---")
print(f"1. Decision Tree (Full): {accuracy_dt_full:.4f}")
print(f"2. Decision Tree (Depth=3): {accuracy_dt_limited:.4f}")
print(f"3. Random Forest: {accuracy_rf:.4f}")


# ----------------------------------------------------------------------
# --- 6. Interpret Feature Importances (Objective 4) ---
# ----------------------------------------------------------------------
print("\n--- Feature Importances (Random Forest) ---")
importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Visualize Feature Importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='teal')
plt.xticks(rotation=45, ha='right')
plt.title('Random Forest Feature Importances')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('03_rf_feature_importances.png')


# ----------------------------------------------------------------------
# --- 7. Evaluate using Cross-Validation (Objective 5) ---
# ----------------------------------------------------------------------
print("\n--- Cross-Validation Evaluation (10-Fold) ---")

# Evaluate the Limited Decision Tree
dt_scores = cross_val_score(dt_classifier_limited, X, y, cv=10, scoring='accuracy')
print(f"Decision Tree (Depth=3) CV Mean Accuracy: {np.mean(dt_scores):.4f}")

# Evaluate the Random Forest
rf_scores = cross_val_score(rf_classifier, X, y, cv=10, scoring='accuracy')
print(f"Random Forest CV Mean Accuracy: {np.mean(rf_scores):.