# Review Preparation: Demo Script and Likely Questions

## Before the review (checklist)
- [ ] Private GitHub repo shared with the faculty (and TAs if assigned); README placeholders filled in
- [ ] `heart.csv` placed next to the notebook; notebook opens and *Restart & Run All* works (skip the Colab `files.upload()` cell if running locally)
- [ ] Model files (`final_elastic_net_model.pkl`, `feature_scaler.pkl`, `feature_names.pkl`) already generated, so the demo does not depend on re-training
- [ ] Slide deck opens offline (download a PDF/PPTX copy as backup)
- [ ] Each teammate can explain the whole pipeline, not just their own part (individual contribution is marked)

## 10-minute demo flow
1. **Problem and data (1 min):** 1,025 rows, 723 duplicates, 302 unique patients.
2. **Pipeline (2 min):** clean, scale, PCA, clustering, stratified split, nine models, GridSearchCV.
3. **Results (3 min):** model table, ROC curves, tuned results, final confusion matrix.
4. **Live prediction (3 min):** run the last cells of the notebook: test-set patient (36.64 %, class 0, correct) and custom patient (60.85 %, class 1). Change one input and re-run.
5. **Conclusions (1 min):** AUROC 0.881, recall 0.879, limitations.

## Likely questions and suggested answers

**Why did you remove 723 rows?**
They were exact duplicate records. Keeping them would let identical patients appear in both train and test sets and inflate the scores.

**Why AUROC for selection instead of accuracy?**
AUROC does not depend on a single 0.5 threshold and ranks models by how well they separate the classes. It is also more stable than accuracy on a small test set.

**Why did you pick Elastic Net if Logistic Regression has the same AUROC?**
They tie on test AUROC (0.881) and CV AUROC (0.917). Elastic Net had slightly higher recall (0.879 vs 0.848), which matters for screening. Be upfront that `l1_ratio = 0` makes it equivalent to L2-regularised logistic regression.

**The SVM had the best CV AUROC (0.923). Why not use it?**
It scored worse on the test set (AUROC 0.844, precision 0.696). That gap suggests variance on a small dataset, and a simpler, more stable model is preferable.

**Is 80 % accuracy good?**
It is reasonable for this dataset, but the test set has only 61 patients: one patient is about 1.6 percentage points. Differences of 1-2 points between models are not statistically meaningful.

**Why did DBSCAN fail?**
In 2-D PCA space (33 % variance retained) the two classes overlap and have similar density. DBSCAN finds density gaps, which do not exist here, so it returns one cluster plus noise (ARI 0.012). K-Means, which splits by distance to centroids, does better (ARI 0.377).

**Why standardise the features?**
SVM, logistic regression, the neural network, PCA and K-Means are scale-sensitive. The scaler was fitted on the training set only to avoid leakage.

**Is the target really myocardial infarction?**
The label denotes presence of heart disease in the source dataset; we use it as a proxy for MI risk. We state this limitation in the write-up.

**What would you do with more time?**
Repeated or nested cross-validation, probability calibration, feature-importance / SHAP analysis, and validation on an external dataset.

## Facts to have at your fingertips
| Item | Value |
|---|---|
| Split | 80/20 stratified, `random_state = 42` → 241 train / 61 test |
| PCA | 2 components, 33.2 % variance |
| K-Means (k = 2) | ARI 0.377, silhouette 0.423 |
| DBSCAN (eps 0.5) | 1 cluster, 27 noise points, ARI 0.012 |
| Final model | Tuned Elastic Net, C = 0.01, l1_ratio = 0 |
| Final metrics | Acc 0.803, Prec 0.784, Recall 0.879, F1 0.829, AUROC 0.881 |
| Confusion matrix | TN 20, FP 8, FN 4, TP 29 |
