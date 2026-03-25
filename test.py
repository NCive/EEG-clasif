
import joblib
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd


dataset = "output_parquet_files"

df_list = []

for file in os.listdir(dataset):
    if file.endswith(".parquet"):
        file_path = os.path.join(dataset, file)
        df = pd.read_parquet(file_path)
        df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)

# print("Columns:", df_all.columns.tolist())


def to_3class(series):
    return pd.qcut(series, q=3, labels=False, duplicates='drop').astype(int)

df_all["Valence_cls"] = to_3class(df_all["Valence"])
df_all["Arousal_cls"] = to_3class(df_all["Arousal"])
df_all["Dominance_cls"] = to_3class(df_all["Dominance"])
df_all["Liking_cls"] = to_3class(df_all["Liking"])


df_all = df_all.dropna()


label_keywords = ["Valence", "Arousal", "Dominance", "Liking"]

feature_cols = [
    col for col in df_all.columns
    if any(stat in col for stat in ["mean", "std", "min", "max"])
    and not any(lbl in col for lbl in label_keywords)
]

print("Number of features:", len(feature_cols))


X = df_all[feature_cols]
scaler = joblib.load("scaler.pkl")  
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

y = df_all[[
    "Valence_cls",
    "Arousal_cls",
    "Dominance_cls",
    "Liking_cls"
]]


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)


idx = int(input(f"Select test sample index from 0 to {len(X_test) - 1}: "))

X_sample = X_test.iloc[[idx]]
y_true = y_test.iloc[idx].values   

model = joblib.load("lgbm_model.pkl")
y_pred = model.predict(X_sample)

label_map = {0: "Low", 1: "Medium", 2: "High"}
emotions = ["Valence", "Arousal", "Dominance", "Liking"]

print(f"\nSample index: {idx}\n")

for i, emotion in enumerate(emotions):
    pred_label = label_map[y_pred[0][i]]
    true_label = label_map[y_true[i]]
    
    print(f"{emotion}: Predicted = {pred_label} | Actual = {true_label}")