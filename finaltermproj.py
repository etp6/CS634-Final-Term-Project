import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

kf = KFold(n_splits=10, shuffle=True, random_state=42)
data = pd.read_csv("breast_cancer_data.csv")
X_df = data.drop(columns=["Diagnosis"])
y_df = data["Diagnosis"]
X_np = X_df.values
y_np = y_df.values

# 'run_random_forest' -- function to run the Random Forest algorithm
def run_random_forest():
    print("Running Random Forest...")
    results = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X_df), start=1):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y_df.iloc[train_idx], y_df.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results.append(compute_metrics(i, tp, tn, fp, fn))

    df = pd.DataFrame(results)
    df.loc["Average"] = df.mean(numeric_only=True)
    print("\nRandom Forest Results:\n", df)
    df.to_csv("rf_results.csv", index=False)

# 'run_knn' -- function to run the K-Nearest Neighbor algorithm
def run_knn():
    print("\nRunning K-Nearest Neighbor (KNN)...")
    results = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X_df), start=1):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y_df.iloc[train_idx], y_df.iloc[test_idx]

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results.append(compute_metrics(i, tp, tn, fp, fn))

    df = pd.DataFrame(results)
    df.loc["Average"] = df.mean(numeric_only=True)
    print("\nKNN Results:\n", df)
    df.to_csv("knn_results.csv", index=False)

# 'run_lstm' -- function to run the Long Short-Term Memory (LSTM) algorithm
def run_lstm():
    print("\nRunning Long Short-Term Memory (LSTM)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    results = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X_lstm), start=1):
        X_train, X_test = X_lstm[train_idx], X_lstm[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        model = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

        y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results.append(compute_metrics(i, tp, tn, fp, fn))

    df = pd.DataFrame(results)
    df.loc["Average"] = df.mean(numeric_only=True)
    print("\nLSTM Results:\n", df)
    df.to_csv("lstm_results.csv", index=False)

# 'compute_metrics' -- function to manually compute metrics
def compute_metrics(fold, tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    tss = tpr + tnr - 1
    hss = 2 * (tp * tn - fn * fp) / (
        (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    ) if ((tp + fn)*(fn + tn) + (tp + fp)*(fp + tn)) else 0

    return {
        "Fold": fold, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "TSS": tss, "HSS": hss
    }

# 'summarize_results' -- function to print final summary/averages of metrics
def summarize_results():
    rf = pd.read_csv("rf_results.csv").tail(1).copy()
    knn = pd.read_csv("knn_results.csv").tail(1).copy()
    lstm = pd.read_csv("lstm_results.csv").tail(1).copy()

    rf["Model"] = "RF"
    knn["Model"] = "KNN"
    lstm["Model"] = "LSTM"

    summary = pd.concat([rf, knn, lstm], ignore_index=True)
    summary = summary[["Model"] + [col for col in summary.columns if col != "Model"]]

    print("\nFinal Summary of Metrics:\n")
    print(summary.round(4))

def main():
    run_random_forest()
    run_knn()
    run_lstm()
    summarize_results()

if __name__ == "__main__":
    main()