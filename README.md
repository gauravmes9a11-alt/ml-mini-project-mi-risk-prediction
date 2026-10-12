# Myocardial Infarction Risk Prediction Using Machine Learning

**UE24CS352A – Machine Learning | Mini-Project**
PES University, Bengaluru

| | |
|---|---|
| **Team** | `<Team No.>` – Gaurav S. (PES1UG24AM104) and `<Teammate name (USN)>` |
| **Problem statement** | `<Serial No. and title as per the master list>` |
| **Section / Faculty** | `<Section>` / `<Faculty name>` |

---

## 1. Overview

This project builds an end-to-end machine learning pipeline that predicts whether a patient is at higher risk of heart disease / myocardial infarction (MI) from 13 routine clinical measurements. It covers:

1. Data cleaning and exploratory data analysis (EDA)
2. Dimensionality reduction (PCA) and unsupervised clustering (K-Means, DBSCAN)
3. Training and comparison of nine supervised classifiers
4. Hyperparameter tuning with 5-fold stratified cross-validation
5. A final risk-prediction function that returns a predicted class and the model's estimated probability

> **Disclaimer:** this is an academic demonstration. It is not a medical device and its output is not a medical diagnosis.

## 2. Dataset

| Property | Value |
|---|---|
| Source | Heart Attack Analysis & Prediction (heart disease) dataset, `heart.csv` |
| Raw size | 1,025 rows × 14 columns |
| After removing duplicates | **302 unique patients** (723 duplicate rows removed) |
| Missing values | None |
| Target | `target` – 1 = heart disease / higher risk (164 patients, 54.3 %), 0 = lower risk (138 patients, 45.7 %) |

**Features (13):** `age`, `sex`, `cp` (chest-pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol), `fbs` (fasting blood sugar > 120 mg/dl), `restecg` (resting ECG result), `thalach` (maximum heart rate), `exang` (exercise-induced angina), `oldpeak` (ST depression), `slope` (ST-segment slope), `ca` (major vessels coloured by fluoroscopy), `thal` (thalassemia test result).

The dataset file is **not** committed to this repository. Place `heart.csv` in the `notebooks/` folder before running (see Section 5).

## 3. Approach

```
heart.csv → de-duplicate → EDA → standardise → PCA (2 components) → K-Means / DBSCAN
                         ↓
           stratified 80/20 split → train 9 classifiers → compare (Acc, Prec, Rec, F1, AUROC)
                         ↓
           GridSearchCV (5-fold stratified, scoring = AUROC) → final model → risk prediction
```

| Stage | Details |
|---|---|
| Cleaning | Duplicate removal, missing-value check |
| EDA | Target balance, histograms and box plots of the 5 numeric features, correlation heat-map |
| Scaling | `StandardScaler` (fitted on the training set only for supervised models) |
| Unsupervised | PCA to 2 components (33.2 % variance retained); K-Means (k = 2…10) and DBSCAN (eps sweep 0.2–2.0), evaluated with Adjusted Rand Index and silhouette score |
| Split | 80 / 20 stratified, `random_state = 42` → 241 train / 61 test |
| Models | Logistic Regression, Elastic Net, SVM (linear, RBF, polynomial), Random Forest, Gradient Boosting, Bagging, Neural Network (MLP 32-16) |
| Tuning | `GridSearchCV` on SVM-polynomial, Elastic Net and Logistic Regression, 5-fold stratified CV, AUROC |
| Final model | Tuned Elastic Net logistic regression (`C = 0.01`, `l1_ratio = 0.0`) |

## 4. Results

### Unsupervised learning
| Method | Result |
|---|---|
| K-Means (k = 2) | ARI = **0.377**, silhouette = 0.423 – clusters partly align with the disease label |
| DBSCAN (eps = 0.5, min_samples = 5) | 1 cluster + 27 noise points, ARI = 0.012 – density-based clustering does not separate the classes |

### Supervised learning (untuned models, 61-patient hold-out set)
| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| Elastic Net | 0.8033 | 0.8000 | 0.8485 | 0.8235 | **0.8723** |
| Logistic Regression | 0.8033 | 0.8000 | 0.8485 | 0.8235 | 0.8712 |
| SVM Polynomial | 0.8197 | 0.7895 | 0.9091 | **0.8451** | 0.8701 |
| Random Forest | 0.7541 | 0.7647 | 0.7879 | 0.7761 | 0.8701 |
| SVM Linear | 0.8197 | 0.8056 | 0.8788 | 0.8406 | 0.8690 |
| SVM RBF | 0.7705 | 0.7714 | 0.8182 | 0.7941 | 0.8420 |
| Bagging | 0.7705 | 0.7714 | 0.8182 | 0.7941 | 0.8371 |
| Neural Network | 0.7541 | 0.7647 | 0.7879 | 0.7761 | 0.8323 |
| Gradient Boosting | 0.7213 | 0.7353 | 0.7576 | 0.7463 | 0.8258 |

### After hyperparameter tuning
| Model | CV AUROC | Test Acc. | Test Recall | Test F1 | Test AUROC |
|---|---|---|---|---|---|
| Tuned Elastic Net (**final**) | 0.917 | 0.8033 | 0.8788 | 0.8286 | **0.881** |
| Tuned Logistic Regression | 0.917 | 0.8033 | 0.8485 | 0.8235 | 0.881 |
| Tuned SVM Polynomial | 0.923 | 0.7541 | 0.9697 | 0.8101 | 0.844 |

**Final model confusion matrix** (test set): TN = 20, FP = 8, FN = 4, TP = 29 → the model catches 29 of 33 high-risk patients (recall 0.879).

### Honest interpretation
- The test set has only 61 patients, so one patient shifts accuracy by about 1.6 percentage points. Differences of 0.01–0.02 between the top models are within noise.
- Tuned Elastic Net and tuned Logistic Regression tie on AUROC (0.881). The selected `l1_ratio = 0.0` means the Elastic Net reduces to an L2-regularised logistic regression. It was kept as the final model for its slightly higher recall, which matters for a screening task.
- The target label denotes the presence of heart disease in the source data; it is used here as a proxy for MI risk.

## 5. Setup and Usage

**Requirements:** Python 3.9+ and Jupyter.

```bash
# 1. Clone and enter the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset next to the notebook
cp /path/to/heart.csv notebooks/heart.csv

# 5. Launch the notebook
cd notebooks
jupyter notebook Myocardial_Infarction_ML_Project.ipynb
```

**Running locally:** the notebook was developed in Google Colab. When running locally, **skip the cell containing `from google.colab import files` / `files.upload()`** – `heart.csv` is already read from the notebook folder. Then run the remaining cells top to bottom (*Kernel → Restart & Run All* after skipping that cell).

**Running in Google Colab:** open the notebook, run all cells, and upload `heart.csv` when prompted.

**Files produced when the notebook runs** (written to the notebook's working folder):

| File | Purpose |
|---|---|
| `heart_cleaned.csv` | De-duplicated dataset |
| `final_elastic_net_model.pkl`, `feature_scaler.pkl`, `feature_names.pkl` | Saved final model, scaler and feature order |
| `all_model_results.csv`, `tuned_model_results.csv`, `final_model_metrics.csv` | Result tables |

The `results/` folder in this repository already contains the three result tables for reference.

### Making a prediction (live demo)
The last section of the notebook defines `show_prediction(...)`, which loads the saved model and scaler and prints the predicted class and probability:

```python
show_prediction(
    age=55, sex=1, cp=1, trestbps=130, chol=240, fbs=0,
    restecg=1, thalach=150, exang=0, oldpeak=1.0,
    slope=1, ca=0, thal=2
)
# Predicted Class : 1
# Probability     : 60.85%
# Model Output    : Higher predicted risk
```

## 6. Repository Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   └── Myocardial_Infarction_ML_Project.ipynb   # complete pipeline
├── data/                                        # place heart.csv here if preferred
├── models/                                      # saved .pkl files (optional)
├── results/                                     # metric tables (CSV)
├── figures/                                     # PCA, clustering, ROC, confusion matrix, etc.
└── docs/                                        # write-up (PDF) and presentation
```

## 7. Limitations and Future Work

- Small dataset (302 unique patients): use repeated or nested cross-validation for more reliable model comparison.
- Class labels reflect heart-disease presence, not confirmed MI events.
- Possible extensions: calibration of predicted probabilities, feature-importance / SHAP analysis, external validation on a larger cohort, and a simple web interface for the predictor.

## 8. Team Contributions

| Member | Contribution |
|---|---|
| Gaurav S. | `<fill in – e.g. EDA, PCA/clustering, model training, tuning>` |
| `<Teammate>` | `<fill in>` |

## 9. License / Academic Use

Prepared as coursework for UE24CS352A, PES University. Dataset credit belongs to its original source.
