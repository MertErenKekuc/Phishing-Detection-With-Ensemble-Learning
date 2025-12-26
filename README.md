# Phishing Detection Using Ensemble Learning

This repository presents a machine learning–based system for detecting phishing websites using URL and website behavior features. The study focuses on evaluating multiple classification algorithms and improving detection performance through ensemble learning techniques.

---

## 📌 Project Overview

Phishing attacks are one of the most common cybersecurity threats targeting internet users. In this project, a supervised machine learning approach is applied to classify websites as **legitimate** or **phishing** based on extracted features.

The project compares individual classifiers with ensemble methods and demonstrates that ensemble learning significantly improves robustness and performance.

---

## 🧠 Machine Learning Models Used

The following models are implemented and evaluated:

- Decision Tree (Entropy-based)
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- Multi-Layer Perceptron (MLP)
- Voting Classifier (Hard Voting)
- **Weighted Soft Voting Ensemble (RF + MLP + XGBoost)**

---

## ⚙️ Data Preprocessing

- Duplicate records are removed
- Missing values are handled using median imputation
- Feature scaling is applied using **Min-Max Normalization**
- The problem is formulated as a **binary classification task**

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC–AUC
- Confusion Matrix

---

## 📈 Experimental Results

### 🔥 Model Performance Comparison
<img width="1000" height="600" alt="model_comparison_detailed" src="https://github.com/user-attachments/assets/719c341c-f1ca-4347-9eec-55f7e62725fb" />

---

### 📉 ROC Curves of All Models
<img width="800" height="600" alt="roc_curves_all" src="https://github.com/user-attachments/assets/e1ec3495-49d4-4d42-9b8f-ee631c7d9982" />

---

### 🧩 Confusion Matrices (Top Performing Models)
<img width="1500" height="1200" alt="confusion_matrices_top4" src="https://github.com/user-attachments/assets/bb56a23b-932d-4fba-a924-35486856aa8d" />

---

### 🔗 Feature Correlation Analysis
<img width="722" height="603" alt="Ekran görüntüsü 2025-12-18 013824" src="https://github.com/user-attachments/assets/8737ed90-f0a3-4475-be78-60e6f2075a4d" />

---

## 🏆 Best Model

The **Weighted Soft Voting Ensemble** achieved the highest overall performance by combining the strengths of Random Forest, MLP, and XGBoost classifiers with different contribution weights.

---

## 🚀 Technologies Used

- Python
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn
- Flask

---

## 🌐 Web Application

In addition to offline model evaluation, a web-based phishing detection application has been developed to demonstrate real-world usability of the proposed system.

The web application allows users to input a URL and receive an instant prediction indicating whether the website is **legitimate** or **phishing** based on the trained machine learning models.

### Application Features
- User-friendly web interface
- Real-time URL phishing prediction
- Backend powered by trained machine learning models
- Demonstrates practical deployment of the proposed approach

This web application shows that the trained models are not only effective in experimental settings but also suitable for real-time cybersecurity applications.

> The web application was developed for demonstration and academic purposes.

<img width="1917" height="988" alt="Ekran görüntüsü 2025-12-18 131911" src="https://github.com/user-attachments/assets/d9f91d20-b29c-418b-bddf-dce02aace68a" />

---

## 📌 Conclusion

Experimental results demonstrate that ensemble learning methods outperform individual classifiers in phishing detection tasks. The proposed weighted soft voting approach provides a robust and effective solution for detecting phishing websites and can be extended for real-world cybersecurity applications.

---

## 📄 License

This project is developed for academic and educational purposes.
