# Breast Cancer Diagnosis

An interactive **Data Analytics and Machine Learning** application for exploring breast tumor data, performing exploratory data analysis (EDA), visualizing feature distributions, comparing machine-learning models, and generating Benign / Malignant predictions — all in a clean Streamlit dashboard.

---

## Project Description

This project applies standard data analytics and machine-learning techniques to the Breast Cancer Wisconsin (Diagnostic) Dataset. It covers the complete analytical pipeline: data loading, cleaning, EDA, feature analysis, model training, evaluation, and prediction — presented through an interactive multi-page web application.

> **Educational project only.** Predictions are not medical diagnoses and must not be used to guide clinical decisions.

---

## Objectives

- Explore and understand the dataset structure and quality
- Perform data cleaning and preprocessing
- Conduct exploratory data analysis with visualizations
- Analyze feature distributions and correlations
- Identify features most associated with diagnosis
- Train and compare machine-learning classification models
- Evaluate models with multiple metrics
- Provide an interactive prediction interface

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

| Field | Value |
|---|---|
| Source | [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) / UCI ML Repository |
| File | `data.csv` |
| Records | 569 patient samples |
| Numeric features | 30 (10 measurements × mean, SE, worst) |
| Target column | `diagnosis` |
| Classes | `B` = Benign (357 cases, 62.7%) · `M` = Malignant (212 cases, 37.3%) |
| Missing values | None in feature columns |

**Feature groups:** For each of 10 measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), three statistics are computed: mean, standard error (SE), and worst (largest value).

---

## Project Structure

```
IBM Project/
│
├── app.py                          ← Streamlit application (main entry point)
├── utils.py                        ← Shared data/model/plot helpers
├── train_models.py                 ← Model training and saving script
├── data.csv                        ← Dataset (download from Kaggle)
│
├── models/
│   ├── logistic_regression_pipeline.pkl
│   ├── decision_tree_pipeline.pkl
│   ├── random_forest_pipeline.pkl
│   ├── metrics.pkl
│   ├── rf_importances.pkl
│   └── feature_cols.pkl
│
├── ProjectReport.docx              ← Full project report with charts
├── Ruthvika_BreastCancerAnalytics.ipynb  ← Jupyter notebook
├── requirements.txt                ← Python dependencies
└── README.md                       ← This file
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3 | Core language |
| Streamlit | Interactive web application |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models, pipelines, metrics |
| Matplotlib | Plotting |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |
| Jupyter | Notebook-based analysis |

---

## Application Pages

The Streamlit app has seven pages, accessible from the left sidebar:

| Page | What it shows |
|---|---|
| 🏠 **Home** | KPI cards (records, features, class counts), distribution chart, workflow, tech stack |
| 📊 **Dataset Explorer** | Data preview, statistics, missing values, data types, class distribution, CSV upload |
| 📈 **Exploratory Analysis** | Distribution tab, Comparison tab (boxplots), Correlation heatmap, Relationships (scatterplots) |
| 🔬 **Feature Insights** | Per-feature statistics, violin plots, Random Forest importance chart (top 15) |
| 🤖 **Model Performance** | Comparison table, model cards, confusion matrices, ROC curves |
| 🔮 **Prediction** | 30-input form (Mean / SE / Worst groups), model selector, probability output |
| ℹ️ **About Project** | Overview, actual results, key findings, limitations |

---

## Data Preprocessing

Steps applied before model training:

1. **Removed** `Unnamed: 32` — entirely null CSV artifact column
2. **Excluded** `id` — patient identifier, no predictive value
3. **Encoded** target: `M` (Malignant) = 1, `B` (Benign) = 0
4. **Verified** zero missing values in all 30 feature columns
5. **Split** 80% train / 20% test with stratification (`random_state=42`)
6. **Scaled** features using `StandardScaler` inside scikit-learn Pipelines — fitted only on training data to prevent data leakage

---

## Exploratory Data Analysis

Charts generated from the actual dataset:

- **Class distribution** — bar chart and pie chart
- **KDE distributions** — per-feature kernel density plots split by diagnosis
- **Boxplots** — Benign vs Malignant comparison for key features
- **Correlation heatmap** — Pearson correlation for all 30 features
- **Scatterplots** — feature pair relationships colored by diagnosis
- **Violin plots** — per-feature distribution in Feature Insights page
- **Group statistics** — mean, median, std per diagnosis class

---

## Machine Learning Models

Three classifiers trained using scikit-learn Pipelines:

| Model | Notes |
|---|---|
| **Logistic Regression** | Linear binary classifier; interpretable; requires scaling (`max_iter=10000`) |
| **Decision Tree** | Recursive partitioning; interpretable; depth-limited (`max_depth=5`) |
| **Random Forest** | Ensemble of 100 trees; reduces overfitting; provides feature importance |

All pipelines chain `StandardScaler → Classifier` to ensure no data leakage.

---

## Model Evaluation

Each model is evaluated on the held-out 20% test set. Results are computed at runtime from the actual dataset — not hard-coded.

**Metrics used:**

| Metric | Description |
|---|---|
| Accuracy | Overall % of correct predictions |
| Precision | Of predicted Malignant, what % are truly Malignant |
| Recall | Of actual Malignant, what % were correctly identified |
| F1-Score | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under the ROC curve (1.0 = perfect, 0.5 = random) |
| Confusion Matrix | TP / TN / FP / FN counts |

> **Recall is particularly important** in this context — a false negative (missed malignant case) is the most consequential type of error.

---

## CSV Upload

The **Dataset Explorer** page includes an optional CSV upload feature. When a file is uploaded:

- The app validates that a `diagnosis` column exists with `M` / `B` values
- Checks for at least one numeric feature column
- Drops the `Unnamed: 32` artifact if present
- Replaces the default dataset for that session's Dataset Explorer and EDA pages
- Shows a clear validation error if the file is incompatible — no crashes

Clearing the upload reverts to the default `data.csv`.

---

## Setup

### 1. Download the dataset

Go to https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data, download `data.csv`, and place it in the project root folder (same directory as `app.py`).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train and save models (run once)

```bash
python train_models.py
```

This generates the `.pkl` files in the `models/` folder. Skip this step if the `models/` folder already contains the trained files.

---

## Run

```bash
streamlit run app.py
```

---

## Localhost

The app opens automatically at:

```
http://localhost:8501
```

---

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Ensure `data.csv` is in the project root folder
3. Run `python train_models.py` if `models/` is empty
4. Start the app: `streamlit run app.py`
5. Open `http://localhost:8501` in your browser
6. Use the sidebar to navigate between pages
7. Explore data in **Dataset Explorer** and **Exploratory Analysis**
8. Review model results in **Model Performance**
9. Enter tumor measurements in **Prediction** to get a classification result

---

## Machine Learning Workflow

```
Dataset (data.csv)
       ↓
Data Cleaning
(remove artifact column, exclude id)
       ↓
Exploratory Data Analysis
(distributions, correlations, comparisons)
       ↓
Preprocessing
(encode target, StandardScaler in Pipeline)
       ↓
Train / Test Split
(80% train, 20% test, stratified)
       ↓
Model Training
(Logistic Regression, Decision Tree, Random Forest)
       ↓
Evaluation
(Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
       ↓
Prediction
(interactive form → loaded pipeline → class + probability)
```

---

## Limitations

- Small dataset (569 samples) from a single institution — results may not generalize broadly
- Binary classification only; real clinical diagnosis involves many additional factors
- No external validation on an independent dataset
- Models were validated on an 80/20 split of the same dataset only
- No clinical review by medical professionals

---

## Disclaimer

This project is developed for **educational and data-analytics purposes only**. The machine-learning models and findings presented here are not validated for clinical use, have not been reviewed by medical professionals, and **must not be used to guide any actual medical decisions**. Always consult qualified healthcare professionals for medical concerns.

---

## References

1. Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set.* UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
2. Kaggle Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. https://scikit-learn.org
4. Streamlit Documentation: https://docs.streamlit.io
5. Waskom, M. (2021). seaborn: statistical data visualization. *JOSS*, 6(60), 3021.
