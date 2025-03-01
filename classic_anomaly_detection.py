import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(file_path):
    """
    Loads the dataset from a CSV file and returns a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df_clean = df.dropna() # Drop missing values (if any exist)
    return df_clean

def normalize_data(df):
    """
    Scales numerical columns (excluding the target column) using StandardScaler.
    Returns the scaled features (X) and labels (y).
    """
    # Separate features and target
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def train_oneclass_svm(X_train):
    """
    Trains a One-Class SVM on normal transactions only, with a progress update.
    """
    print("Training One-Class SVM... This may take a while.")
    
    # Initialize progress bar
    with tqdm(total=1, desc="Training", unit="step") as pbar:
        oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.001)
        oc_svm.fit(X_train)  # This is where it trains
        pbar.update(1)  # Manually update progress

    print("Training completed.")
    return oc_svm


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained One-Class SVM model on test data containing
    both normal and fraudulent transactions. Prints various performance metrics.
    """
    # Get predictions: +1 for inliers, -1 for outliers
    y_pred = model.predict(X_test)

    # Convert +1 -> 0 (normal), -1 -> 1 (fraud)
    y_pred = np.where(y_pred == 1, 0, 1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("Confusion Matrix:")
    print(cm)

    return y_pred

def main():
    df = load_data('/home/karam-abu-judom/Downloads/financial_transaction_dataset/creditcard.csv')
    df = clean_data(df)
    X, y = normalize_data(df)

    # Split the dataset so that we can train on normal data only. This is how one-class SVM works, but we'll keep a test set that includes both normal and fraud.
    # We isolate normal transactions (class=0) for training.
    X_train_full = X[y == 0]
    X_train, X_val = train_test_split(X_train_full, test_size=0.2, random_state=42)

    oc_svm_model = train_oneclass_svm(X_train) # Training the model

    # Evaluate on a test set that includes both normal & fraud from the entire dataset.
    # Typically, you'd keep a separate set aside, but for simplicity, use an 80/20 split on the entire dataset.
    X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = evaluate_model(oc_svm_model, X_test, y_test)

    # Identify anomalies (fraudulent predictions) in the final test set
    anomalies = X_test[y_pred == 1]
    print(f"\nNumber of detected anomalies (fraud): {len(anomalies)}")

    # Optionally, we can store or view the anomalous points alongside their actual labels
    anomalies_df = pd.DataFrame(anomalies, columns=df.drop('Class', axis=1).columns)
    anomalies_df['Predicted_Class'] = 1
    # print("\nSample anomalies:")
    # print(anomalies_df.head())

if __name__ == "__main__":
    main()
