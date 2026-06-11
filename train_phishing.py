import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import os as _os

# ── Paths ─────────────────────────────────────────────────────────────────────
# DATASET_PATH  → your CSV (the one from Kaggle)
# MODEL_PATH    → where the trained model will be saved (backend/models/)
# SCALER_PATH   → where the scaler will be saved (backend/models/)
# Change DATASET_PATH if your CSV is somewhere else.

DATASET_PATH = r"D:\cyber\kaggle\Dataset.csv"

_HERE             = _os.path.dirname(_os.path.abspath(__file__))
MODEL_PATH        = _os.path.join(_HERE, "..", "models", "model.joblib")
SCALER_PATH       = _os.path.join(_HERE, "..", "models", "scaler.joblib")
CONF_MATRIX_IMAGE = _os.path.join(_HERE, "..", "models", "confusion_matrix.png")



def train_mlp_model():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # Handle missing values
    print("Handling missing values...")
    missing_tld_count = df['tld'].isnull().sum()
    print(f"Found {missing_tld_count} missing values in 'tld'. Filling with empty string.")
    df['tld'] = df['tld'].fillna('')

    def keyword_mismatch(dom, tld, subdomain="", path=""):
        edu_gov_keywords = ['nios','cbse','upsc','ssc','railway','passport','neet','ugc','aicte','ncert']
        trusted_suffixes = ['gov.in','nic.in','ac.in','edu.in']
        raw = f"{subdomain} {dom} {path}".lower().replace('-', '').replace('.', '')
        for kw in edu_gov_keywords:
            if kw in raw and tld not in trusted_suffixes:
                return 1
        return 0

    df['keyword_mismatch'] = df.apply(lambda row: keyword_mismatch(row['dom'], row['tld'], row.get('subdomain', ''), row.get('path', '')),
                                      axis=1)

    
    # 23 numerical features used for training
    feature_cols = [
        'url_len', 'dom_len', 'is_ip', 'tld_len', 'subdom_cnt', 'letter_cnt', 'digit_cnt',
        'special_cnt', 'eq_cnt', 'qm_cnt', 'amp_cnt', 'dot_cnt', 'dash_cnt', 'under_cnt',
        'letter_ratio', 'digit_ratio', 'spec_ratio', 'is_https', 'slash_cnt', 'entropy',
        'path_len', 'query_len', 'keyword_mismatch'
    ]
    print("Training feature columns:", feature_cols)

    
    X = df[feature_cols]
    y = df['label']
    
    # Split dataset into training and testing (80% train, 20% test)
    print("Splitting dataset into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling (crucial for MLP convergence)
    print("Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize MLPClassifier
    print("Initializing Multi-Layer Perceptron (MLP) Classifier...")
    
    mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    learning_rate_init=0.001,
    verbose=True
    )

    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    print("Training the MLP model (this may take a couple of minutes)...")
    # mlp.fit(X_train_scaled, y_train)
    mlp.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    print("Model training completed.")
    
    # Predict on test set
    print("Evaluating model on test set...")
    y_pred = mlp.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- MODEL PERFORMANCE REPORT ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Phishing (1)']))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot and save confusion matrix
    print("Generating confusion matrix plot...")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'], 
                yticklabels=['Legitimate', 'Phishing'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('MLP Phishing URL Detector Confusion Matrix')
    plt.tight_layout()
    
    # Save image in project dir
    plt.savefig(CONF_MATRIX_IMAGE, dpi=150)
    print(f"Saved confusion matrix plot to {CONF_MATRIX_IMAGE}")
    

    plt.close()
    
    # Save model and scaler
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(mlp, MODEL_PATH)
    print(f"Saving scaler to {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)
    print("All models and scalers saved successfully!")

if __name__ == "__main__":
    train_mlp_model()
