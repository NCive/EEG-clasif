import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

dataset = "output_parquet_files"

df_list = []

# Load all parquet files
for file in os.listdir(dataset):
    if file.endswith(".parquet"):
        file_path = os.path.join(dataset, file)
        df = pd.read_parquet(file_path)
        df_list.append(df)

# Combine into ONE DataFrame
df_all = pd.concat(df_list, ignore_index=True)

# print("Columns:", df_all.columns.tolist())


# ✅ Convert continuous labels → 3 classes
def to_3class(series):
    return pd.qcut(series, q=3, labels=False, duplicates='drop').astype(int)

df_all["Valence_cls"] = to_3class(df_all["Valence"])
df_all["Arousal_cls"] = to_3class(df_all["Arousal"])
df_all["Dominance_cls"] = to_3class(df_all["Dominance"])
df_all["Liking_cls"] = to_3class(df_all["Liking"])


# ✅ Remove rows with NaNs (important after qcut)
df_all = df_all.dropna()


# ✅ Feature selection (EXCLUDE labels to prevent leakage)
label_keywords = ["Valence", "Arousal", "Dominance", "Liking"]

feature_cols = [
    col for col in df_all.columns
    if any(stat in col for stat in ["mean", "std", "min", "max"])
    and not any(lbl in col for lbl in label_keywords)
]

print("Number of features:", len(feature_cols))


# Features and targets
X = df_all[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df_all[[
    "Valence_cls",
    "Arousal_cls",
    "Dominance_cls",
    "Liking_cls"
]]



X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)


# ✅ LightGBM model
base_model = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=1500,
    learning_rate=0.1
)

# base_model = XGBClassifier(
#     objective="multi:softmax",
#     num_class=3,
#     n_estimators=100,
#     learning_rate=0.05
# )

model = MultiOutputClassifier(base_model)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
for i, col in enumerate(y.columns):
    print(f"\n=== {col} ===")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# Save the model
joblib.dump(model, "lgbm_model.pkl")
joblib.dump(scaler, "scaler.pkl")