# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import evaluate_print
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
    

# Function for loading the dataset
def load_data(file_path):
    
    try:
        data = pd.read_csv(file_path)
        print("Credit Card Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print("Error: Credit Card Dataset file not found.")
        return None

# Function to Preprocess the data
def preprocess_data(data):
    # drop time 
    data.drop(['Time'], axis=1, inplace=True)

    # Separate features and labels
    X = data.drop(columns=['Class'], axis=1)
    y = data['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_autoencoder(X_train_scaled, X_test_scaled, y_train, y_test, contamination=0.0017):
    print("Training AutoEncoder model...")
    # Initialize the AutoEncoder model
    model = AutoEncoder(
        hidden_neuron_list=[64, 32, 32, 64],  
        epoch_num=10,                        
        contamination=contamination,      
        verbose=1
    )
    
    # Train the model
    model.fit(X_train_scaled)
    
    print("AutoEncoder training completed!")
    
    # get the prediction on the test data
    y_test_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)

    # Get the anomaly scores
    train_scores = model.decision_function(X_train_scaled)
    test_scores = model.decision_function(X_test_scaled)
    
    # Evaluate on training data
    print("\nTraining Set Performance:")
    print(f"ROC AUC Score: {roc_auc_score(y_train, train_scores):.4f}")
    evaluate_print('AutoEncoder', y_train, train_scores)
        
    # Evaluate on test data
    print("\nTest Set Performance:")
    print(f"ROC AUC Score: {roc_auc_score(y_test, test_scores):.4f}")
    evaluate_print('AutoEncoder', y_test, test_scores)
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print(conf_matrix)
    
    # Evaluate the model
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Non-Fraud', 'Fraud']))
    
    return test_scores, conf_matrix, roc_auc_score(y_test, test_scores)

def visualize_results(anomaly_scores, y_test, conf_matrix, roc_auc_score):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))  
    # Anomaly curve
    axes[0,0].hist(anomaly_scores[y_test == 0], bins=50, alpha=0.5, label='Non-Fraud', color='blue')
    axes[0,0].hist(anomaly_scores[y_test == 1], bins=50, alpha=0.5, label='Fraud', color='red')
    axes[0,0].set_title('Distribution of Anomaly Scores')
    axes[0,0].set_xlabel('Anomaly Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc_score:.3f})')
    axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0,1].set_xlim([0.0, 1.0])
    axes[0,1].set_ylim([0.0, 1.05])
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend(loc="lower right")
    axes[0,1].grid(True)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)
    avg_precision = average_precision_score(y_test, anomaly_scores)
    
    axes[1,0].plot(recall, precision, color='blue', lw=2,label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision-Recall Curve')
    axes[1,0].legend()
    axes[1,0].grid(True)
        
    # Confusion Matrix Heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1,1].set_title('Confusion Matrix')
    axes[1,1].set_xlabel('Predicted')
    axes[1,1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('output.png')
    plt.show()
    
# Main
def main():
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION SYSTEM USING AUTOENCODER")
    print("="*60)
    
    # File path to the credit card dataset 
    file_path = 'creditcard.csv'
    print("\n1. Loading credit card dataset...")
    # Loads credit card data
    data = load_data(file_path)
    if data is None:
        return
    
    print("\n2. Preprocessing data...")
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test= preprocess_data(data)
    
    print("\n3. Training AutoEncoder model...")
    # Train and evaluate the model
    # Contamination is set to ~0.0017 (492 frauds / 284,807 transactions)
    anomaly_scores, conf_matrix, roc_auc_score = train_autoencoder(X_train_scaled, X_test_scaled, y_train, y_test, contamination=0.0017)
    
    # Visualize results
    print("\n3. Visualize the results...")
    visualize_results(anomaly_scores, y_test, conf_matrix, roc_auc_score)
        
    print(f"\n{'='*60}")
    print("CREDIT CARD FRAUD DETECTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
if __name__ == "__main__":
    main()