# -*- coding: utf-8 -*-

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_PATH = "data/dataset.csv"
MODEL_OUT = "models/phishing_model.pkl"
SCALER_OUT = "models/scaler.pkl"

# Dataset sütun isimleri
URL_FEATURES = [
    "having_IPhaving_IP_Address",
    "URLURL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "port",
    "HTTPS_token",
    "Submitting_to_email",
    "Redirect",
    "Abnormal_URL",
    "popUpWidnow",
    "Iframe",
    "on_mouseover",
    "RightClick",
]

print("="*70)
print("PHISHING DETECTION - WEIGHTED SOFT VOTING MODEL EGITIMI")
print("="*70)

df = pd.read_csv(DATA_PATH)
y = df["Result"]
if -1 in y.unique():
    y = y.replace(-1, 0)

X = df[URL_FEATURES].copy()

print(f"\nVeri Yuklendi:")
print(f"  Toplam ornekler: {len(df)}")
print(f"  Ozellik sayisi: {len(URL_FEATURES)}")
print(f"  Sinif dagilimi:")
print(f"    - Legitimate (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"    - Phishing (1):   {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

# %10 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\nTrain/Test Ayrimi:")
print(f"  Egitim: {len(X_train)} ornek (%{len(X_train)/len(X)*100:.0f})")
print(f"  Test:   {len(X_test)} ornek (%{len(X_test)/len(X)*100:.0f})")

# ============================================================================
# WEIGHTED SOFT VOTING
# ============================================================================
print("\n" + "="*70)
print("WEIGHTED SOFT VOTING ENSEMBLE EGITILIYOR...")
print("="*70)

model = VotingClassifier(
    estimators=[
        # RandomForest (3)
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )),
        
        # MLP (2)
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(200, 150, 100),
            alpha=0.0001,
            max_iter=6000,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate_init=0.001,
            solver='adam',
            random_state=42
        )),
        
        # XGBoost (1)
        ('xgb', XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            reg_alpha=0.01,
            reg_lambda=1.0,
            eval_metric='logloss',
            random_state=42
        ))
    ],
    voting='soft',
    weights=[3, 2, 1],
    n_jobs=-1
)

print("\nModel Yapisi:")
print("  - RandomForest:  Agirlik=3")
print("  - MLP:           Agirlik=2")
print("  - XGBoost:       Agirlik=1")
print("  - Voting Tipi:   Soft (Olasilik tabanli)")

print("\nEgitim basliyor...")
model.fit(X_train_s, y_train)
print("[OK] Egitim tamamlandi!")

y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*70)
print("PERFORMANS RAPORU")
print("="*70)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"\nDetayli:")
print(f"  False Positives: {cm[0][1]}")
print(f"  False Negatives: {cm[1][0]}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)

print("\n" + "="*70)
print("MODEL KAYDETME")
print("="*70)
print(f"Model: {MODEL_OUT}")
print(f"Scaler: {SCALER_OUT}")
print(f"\nModel Detaylari:")
print(f"  Tip:            VotingClassifier (Weighted Soft)")
print(f"  Base Modeller:  RF, MLP, XGBoost")
print(f"  Agirliklar:     [3, 2, 1]")
print(f"  Ozellik Sayisi: {len(URL_FEATURES)}")
print(f"  Test Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"  Test F1-Score:  {f1:.4f}")
print("\n" + "="*70)
print("MODEL HAZIR! app.py")
print("="*70)