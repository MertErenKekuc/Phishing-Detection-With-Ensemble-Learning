# -*- coding: utf-8 -*-
# Phishing Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

print("="*70)
print("PHISHING DETECTION - VERİ ÖN İŞLEME VE MODEL EĞİTİMİ")
print("="*70)

# ============================================================================
# 1. VERİ YÜKLEME
# ============================================================================
print("\n[1/6] Veri Seti Yükleniyor...")
df = pd.read_csv("data/dataset.csv")
print(f" Veri yüklendi. Boyut: {df.shape}")

# ============================================================================
# 2. VERİ TEMİZLEME
# ============================================================================
print("\n[2/6] Veri Temizleme...")

# Tekrar eden kayıtları kaldır
initial_rows = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
removed_duplicates = initial_rows - df.shape[0]
print(f"   Tekrar eden kayıtlar: {removed_duplicates} adet kaldırıldı")

# Index sütununu kaldır
if 'Index' in df.columns:
    df = df.drop(columns=['Index'])
    print("   'Index' sütunu kaldırıldı")

# Eksik değer kontrolü
missing_count = df.isnull().sum().sum()
if missing_count > 0:
    print(f"   Eksik değer tespit edildi: {missing_count} adet")
    # Eksik değerleri median ile doldur
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Result' in numeric_cols:
        numeric_cols.remove('Result')
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    print("   Eksik değerler median ile dolduruldu")
else:
    print("   Eksik değer bulunamadı")

print(f" Temizleme sonrası boyut: {df.shape}")

# ============================================================================
# 3. ÖZELLİK VE ETİKET AYIRMA
# ============================================================================
print("\n[3/6] Özellik ve Etiket Ayrımı...")

# Hedef değişken
y = df['Result']
if -1 in y.unique():
    y = y.replace(-1, 0)
    print("   Etiketler normalize edildi: -1 → 0")

# Sınıf dağılımı
class_dist = y.value_counts()
print(f"  Sınıf Dağılımı:")
print(f"    - Legitimate (0): {class_dist[0]} ({class_dist[0]/len(y)*100:.1f}%)")
print(f"    - Phishing (1):   {class_dist[1]} ({class_dist[1]/len(y)*100:.1f}%)")

# Özellikler
X = df.drop(columns=['Result'], errors='ignore')
print(f" Özellik sayısı: {X.shape[1]}")

# ============================================================================
# 4. NORMALİZASYON
# ============================================================================
print("\n[4/6] Min-Max Normalizasyon...")

# Normalizasyon öncesi istatistikler
print("  Normalizasyon öncesi (örnek 3 özellik):")
for col in X.columns[:3]:
    print(f"    {col}: Min={X[col].min():.2f}, Max={X[col].max():.2f}")

# Min-Max Scaler (0-1 aralığı)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Normalizasyon sonrası kontrol
print("  Normalizasyon sonrası:")
for i, col in enumerate(X.columns[:3]):
    print(f"    {col}: Min={X_scaled[:, i].min():.2f}, Max={X_scaled[:, i].max():.2f}")

print(" Tüm özellikler [0,1] aralığına ölçeklendirildi")

# ============================================================================
# 5. EĞİTİM/TEST AYIRIMI
# ============================================================================
print("\n[5/6] Eğitim ve Test Setleri...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.1,      # %10 test
    stratify=y,         # Sınıf dağılımını korumak için
    random_state=42
)

print(f"   Eğitim seti: {X_train.shape[0]} örnek (%{X_train.shape[0]/len(X)*100:.0f})")
print(f"   Test seti:   {X_test.shape[0]} örnek (%{X_test.shape[0]/len(X)*100:.0f})")

# ============================================================================
# 6. MODEL TANIMLARI
# ============================================================================
print("\n[6/6] Model Tanımlamaları...")

models = {
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), solver='lbfgs', max_iter=3500, random_state=42),
}

# Ensemble Modeller
# Hard Voting
models["Voting"] = VotingClassifier(
    estimators=[
        ('rf', models["Random Forest"]),
        ('xgb', models["XGBoost"]),
        ('mlp', models["MLP"])
    ],
    voting='hard'
)

# Weighted Soft Voting
models["Weighted Soft Voting"] = VotingClassifier(
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

print(f" Toplam {len(models)} model tanımlandı")

# ============================================================================
# 7. MODEL EĞİTİMİ VE DEĞERLENDİRME
# ============================================================================
print("\n" + "="*70)
print("MODEL EĞİTİMİ VE DEĞERLENDİRME")
print("="*70)

results = []
os.makedirs("results/figures", exist_ok=True)

for idx, (name, model) in enumerate(models.items(), 1):
    print(f"\n[{idx}/{len(models)}] {name} eğitiliyor...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    })

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

print("\n" + "="*70)
print("PERFORMANS KARŞILAŞTIRMASI")
print("="*70)
print(results_df.to_string(index=False))

# En iyi model
best = results_df.iloc[0]
print(f"\n EN İYİ MODEL: {best['Model']}")
print(f"   Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)")

# ============================================================================
# 8. GÖRSELLEŞTİRMELER
# ============================================================================
print("\n" + "="*70)
print("GÖRSELLEŞTİRMELER")
print("="*70)

# Korelasyon Matrisi
print("\n[1/5] Korelasyon matrisi...")
plt.figure(figsize=(12, 10))

# Normalize edilmiş veriden korelasyon matrisi oluştur
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
corr_matrix = X_scaled_df.corr()

# En yüksek ortalama korelasyona sahip ilk 20 özellik
mean_corr = corr_matrix.abs().mean().sort_values(ascending=False)
top_features = mean_corr.head(20).index
corr_subset = corr_matrix.loc[top_features, top_features]

sns.heatmap(
    corr_subset,
    annot=False,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    vmin=-1,
    vmax=1
)
plt.title("Özellikler Arası Korelasyon Matrisi (En Yüksek Korelasyonlu 20 Özellik)", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/correlation_matrix.png", dpi=150)
print("   Kaydedildi: correlation_matrix.png")
plt.show()

# Performans Karşılaştırma
print("\n[2/5] Performans grafiği...")
plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=results_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
    x="Model", y="Score", hue="Metric"
)
plt.title("Farklı Algoritmaların Performans Karşılaştırması")
plt.ylim(0.9, 1.0)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", label_type="edge", fontsize=8)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/figures/model_comparison_detailed.png")
print("   Kaydedildi: model_comparison_detailed.png")
plt.show()

# ROC Eğrileri
print("\n[3/5] ROC eğrileri...")
plt.figure(figsize=(8,6))

def _get_score_for_roc(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass
    if hasattr(model, "estimators_"):
        probs = []
        for est in model.estimators_:
            if hasattr(est, "predict_proba"):
                try:
                    probs.append(est.predict_proba(X)[:, 1])
                except Exception:
                    continue
            elif hasattr(est, "decision_function"):
                try:
                    probs.append(est.decision_function(X))
                except Exception:
                    continue
        if len(probs) > 0:
            arr = np.vstack(probs)
            return arr.mean(axis=0)
    return None

for name in results_df["Model"].tolist():
    model = models[name]
    y_score = _get_score_for_roc(model, X_test)
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        if name == "Weighted Soft Voting":
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=3, color='red')
        else:
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Eğrileri (Tüm Modeller)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/roc_curves_all.png")
print("   Kaydedildi: roc_curves_all.png")
plt.show()

# Confusion Matrix
print("\n[4/5] Confusion matrix...")
top4 = results_df.head(4)["Model"].tolist()
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for i, name in enumerate(top4):
    model = models[name]
    y_pred_model = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_model)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[i],
        cbar=False,
        annot_kws={"fontsize": 8},
        linewidths=0.5,
        linecolor="white",
        square=True
    )
    axes[i].set_title(f"Confusion Matrix - {name}", fontsize=10)
    axes[i].set_xlabel("Tahmin Edilen", fontsize=9)
    axes[i].set_ylabel("Gerçek", fontsize=9)
    axes[i].tick_params(axis='both', which='major', labelsize=8)

for j in range(len(top4), 4):
    fig.delaxes(axes[j])

plt.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.35, wspace=0.25)
plt.savefig("results/figures/confusion_matrices_top4.png", dpi=150)
print("   Kaydedildi: confusion_matrices_top4.png")
plt.show()

# Feature Importance
print("\n[5/5] Feature importance...")
if hasattr(models["Random Forest"], "feature_importances_"):
    importances = models["Random Forest"].feature_importances_
    feat_names = X.columns
    imp_df = pd.DataFrame({
        "Feature": feat_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(data=imp_df.head(15), x="Importance", y="Feature", palette="viridis")
    plt.title("En Önemli 15 Özellik (Random Forest)")
    plt.tight_layout()
    plt.savefig("results/figures/feature_importance.png")
    print("   Kaydedildi: feature_importance.png")
    plt.show()

# ============================================================================
# VOTING vs WEIGHTED SOFT VOTING KARŞILAŞTIRMASI
# ============================================================================
print("\n[6/6] Voting vs Weighted Soft Voting karşılaştırması...")

# İki modelin tahminleri
voting_model = models["Voting"]
weighted_model = models["Weighted Soft Voting"]

y_pred_voting = voting_model.predict(X_test)
y_pred_weighted = weighted_model.predict(X_test)

# Confusion Matrix'ler
cm_voting = confusion_matrix(y_test, y_pred_voting)
cm_weighted = confusion_matrix(y_test, y_pred_weighted)

# Metrikler
acc_voting = accuracy_score(y_test, y_pred_voting)
acc_weighted = accuracy_score(y_test, y_pred_weighted)
f1_voting = f1_score(y_test, y_pred_voting)
f1_weighted = f1_score(y_test, y_pred_weighted)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Voting (Hard) Confusion Matrix
sns.heatmap(
    cm_voting,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0],
    cbar=True,
    annot_kws={"fontsize": 14, "weight": "bold"},
    linewidths=2,
    linecolor="white",
    square=True,
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)
axes[0].set_title(
    f"Voting Classifier (Hard)\nAccuracy: {acc_voting:.4f} | F1: {f1_voting:.4f}",
    fontsize=12,
    fontweight='bold',
    pad=15
)
axes[0].set_xlabel("Tahmin Edilen", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Gerçek", fontsize=11, fontweight='bold')
axes[0].tick_params(axis='both', labelsize=10)

# Detaylı istatistikler
tn_v, fp_v, fn_v, tp_v = cm_voting.ravel()
axes[0].text(
    0.5, -0.15,
    f"TN: {tn_v} | FP: {fp_v} | FN: {fn_v} | TP: {tp_v}",
    ha='center',
    transform=axes[0].transAxes,
    fontsize=9,
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
)

# Weighted Soft Voting Confusion Matrix
sns.heatmap(
    cm_weighted,
    annot=True,
    fmt="d",
    cmap="Greens",
    ax=axes[1],
    cbar=True,
    annot_kws={"fontsize": 14, "weight": "bold"},
    linewidths=2,
    linecolor="white",
    square=True,
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)
axes[1].set_title(
    f"Weighted Soft Voting (3:2:1)\nAccuracy: {acc_weighted:.4f} | F1: {f1_weighted:.4f}",
    fontsize=12,
    fontweight='bold',
    pad=15
)
axes[1].set_xlabel("Tahmin Edilen", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Gerçek", fontsize=11, fontweight='bold')
axes[1].tick_params(axis='both', labelsize=10)

tn_w, fp_w, fn_w, tp_w = cm_weighted.ravel()
axes[1].text(
    0.5, -0.15,
    f"TN: {tn_w} | FP: {fp_w} | FN: {fn_w} | TP: {tp_w}",
    ha='center',
    transform=axes[1].transAxes,
    fontsize=9,
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
)

fig.suptitle(
    "Ensemble Modeller Karşılaştırması: Hard vs Soft Voting",
    fontsize=14,
    fontweight='bold',
    y=1.02
)

plt.tight_layout()
plt.savefig("results/figures/voting_comparison.png", dpi=150, bbox_inches='tight')
print("   Kaydedildi: voting_comparison.png")
plt.show()

print("\n" + "="*70)
print("VOTING KARŞILAŞTIRMASI DETAYLI ANALİZ")
print("="*70)
print(f"\n{'Metrik':<25} {'Hard Voting':<20} {'Weighted Soft Voting':<20}")
print("-"*70)
print(f"{'Accuracy':<25} {acc_voting:<20.4f} {acc_weighted:<20.4f}")
print(f"{'F1-Score':<25} {f1_voting:<20.4f} {f1_weighted:<20.4f}")
print(f"{'True Negatives (TN)':<25} {tn_v:<20} {tn_w:<20}")
print(f"{'False Positives (FP)':<25} {fp_v:<20} {fp_w:<20}")
print(f"{'False Negatives (FN)':<25} {fn_v:<20} {fn_w:<20}")
print(f"{'True Positives (TP)':<25} {tp_v:<20} {tp_w:<20}")

print("\n" + "="*70)
print("TÜM İŞLEMLER TAMAMLANDI")
print("="*70)
