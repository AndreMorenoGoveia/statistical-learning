# Adult Census Income - Classification Report

**Author:** André Moreno Goveia - 13682785
**Course:** Statistical Learning  
**Dataset:** UCI Adult (Census Income)  
**Task:** Predict whether a person's annual income exceeds US$50,000

---

## 1. Dataset Overview

The Adult dataset was extracted from the 1994 U.S. Census Bureau database. It contains demographic and employment attributes for 48,842 individuals, pre-split by UCI into 32,561 training instances and 16,281 test instances. The binary target is whether a person earns more than US$50,000 per year.

The 14 predictor features include 6 numeric attributes (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical ones (workclass, education, marital-status, occupation, relationship, race, sex, native-country).

**Class balance.** The dataset is moderately imbalanced: 75.9% of training instances are `<=50K` and 24.1% are `>50K` (test split: 76.4% / 23.6%). A majority-class baseline would reach ~76% accuracy, making accuracy alone insufficient as an evaluation metric.

---

## 2. Data Analysis and Preprocessing

### 2.1 Missing Values

Missing values are encoded as `?`. Three features contain them:

| Feature | Missing count | Rate |
|---|---|---|
| occupation | 1,843 | 5.66% |
| workclass | 1,836 | 5.64% |
| native-country | 583 | 1.79% |

In total, 2,399 rows (7.37%) have at least one missing value. Since all affected features are categorical, missing entries were imputed with the most frequent category, avoiding the loss of a substantial portion of training data.

### 2.2 Feature Selection

Two features were dropped:

- **`fnlwgt`**: a census sampling weight representing how many people a record stands for at the population level. It carries no information about individual income.
- **`education-num`**: a monotone integer encoding of `education`. Keeping both would double-count the same information; the categorical `education` was retained.

After removal, 12 predictors remain: 4 numeric and 8 categorical.

### 2.3 Preprocessing Pipeline

A unified `ColumnTransformer` was applied identically to all classifiers:

- **Numeric** (age, capital-gain, capital-loss, hours-per-week): median imputation → `KBinsDiscretizer` with 5 quantile bins → one-hot encoding. Quantile binning was chosen because capital-gain and capital-loss are strongly zero-inflated (median = 0, max = 99,999), making standard scaling unreliable. Quantile bins distribute observations evenly regardless of raw scale. For the zero-heavy capital features, some bins collapse to a degenerate interval and are automatically removed by scikit-learn.
- **Categorical**: most-frequent imputation → one-hot encoding (`handle_unknown="ignore"` so that country codes only present in the test split do not cause errors).

The resulting feature matrix is a sparse binary matrix with 110 columns.

---

## 3. kNN Classifier

### 3.1 Hyperparameter Selection via Cross-Validation

Two hyperparameters were tuned via 3-fold stratified cross-validation scored by F1 on the minority class, which is more informative than accuracy under class imbalance:

- **`n_neighbors`** ∈ {5, 11, 21, 31, 51}
- **`weights`** ∈ {`uniform`, `distance`}

Top 5 of 10 configurations:

| k | weights | CV F1 (mean ± std) |
|---|---|---|
| **21** | **uniform** | **0.6251 ± 0.0060** |
| 31 | uniform | 0.6207 ± 0.0068 |
| 51 | uniform | 0.6206 ± 0.0058 |
| 11 | uniform | 0.6189 ± 0.0093 |
| 5 | uniform | 0.6089 ± 0.0093 |

All `distance`-weighted configurations ranked below all `uniform` ones. In a 110-dimensional sparse one-hot space, distances become poorly discriminative (curse of dimensionality), so inverse-distance weighting amplifies noise from poorly-informative neighbours. Uniform weighting, which treats all k neighbours equally, is more robust. The trend of larger k performing better is consistent with this: more neighbours reduce the variance of the local vote.

The best configuration is **k = 21, uniform weighting** (CV F1 = 0.6251).

### 3.2 Holdout Evaluation

The selected model was retrained on the full training set and evaluated on `adult.test`:

| | <=50K | >50K | Overall |
|---|---|---|---|
| Precision | 0.8777 | 0.6549 | — |
| Recall | 0.9033 | 0.5931 | — |
| F1 | 0.8903 | 0.6225 | — |
| Accuracy | — | — | **0.8300** |

**Confusion matrix:**

| | Pred <=50K | Pred >50K |
|---|---|---|
| True <=50K | 11,233 | 1,202 |
| True >50K | 1,565 | 2,281 |

The model correctly classifies 83.0% of instances. For the minority class, recall is 0.593: about 41% of true high earners are missed. This is characteristic of kNN under class imbalance, neighbourhoods are dominated by majority-class examples, creating a systematic bias toward `<=50K`.

---

## 4. Comparison with Other Classifiers

### 4.1 Models

The kNN result was compared against two classifiers from distinct learning paradigms:

- **Logistic Regression**: a discriminative linear model that fits a hyperplane in feature space by maximising the log-likelihood of the binary outcome. Convergence tolerance was relaxed to 2,000 iterations given the high-dimensional one-hot input.
- **Random Forest**: an ensemble of 300 decision trees, each trained on a bootstrap sample with random feature subsets. It models non-linear interactions and is generally robust to the choice of hyperparameters.

All three classifiers used the same preprocessing pipeline and holdout set.

### 4.2 Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| kNN (k=21, uniform) | 0.8300 | 0.6549 | **0.5931** | 0.6225 |
| Logistic Regression | **0.8385** | **0.6922** | 0.5694 | **0.6248** |
| Random Forest | 0.8265 | 0.6533 | 0.5655 | 0.6063 |

**Logistic Regression** achieves the best accuracy and F1, with the highest precision (0.692) which makes fewer false positive predictions for the minority class. The tradeoff is the lowest recall (0.569).

**kNN** has the best recall (0.593), which matters in scenarios where missing a genuine high earner is costly. F1 is essentially tied with Logistic Regression (0.622 vs 0.625).

**Random Forest** performs worst on all four metrics. This is likely because pre-discretizing features into 5 bins removes the continuous threshold information that tree ensembles rely on. Each feature effectively provides at most log₂(5) ≈ 2 useful splits, limiting the expressiveness of individual trees. Logistic Regression and kNN operate directly on the resulting binary indicators and are not affected in the same way.

Overall the three models are within a narrow range (0.83–0.84 accuracy), all substantially above the 0.764 majority-class baseline. The small performance differences suggest the bottleneck is the class imbalance rather than the algorithm.

---

## 5. Conclusion

A kNN classifier was built to predict whether a census respondent earns more than US$50,000 per year. After dropping redundant and non-predictive features, missing categoricals were imputed by mode and numeric features were discretized using quantile binning. Cross-validation over 10 configurations selected **k = 21 with uniform weighting**, achieving a holdout accuracy of 83.0% and F1 of 0.622 on the minority class.

Compared to Logistic Regression and Random Forest under identical preprocessing, kNN is competitive: it ties Logistic Regression on F1 and leads on recall, while Random Forest lags on all metrics, a consequence of discretization clipping the continuous signal that tree-based models exploit. For this task, simpler models are at least as effective as the ensemble.
