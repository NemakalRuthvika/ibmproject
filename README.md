# Breast Cancer Diagnosis

An interactive **Data Analytics and Machine Learning** application built using Python and Streamlit to analyze breast tumor data, visualize patterns, compare machine-learning models, and predict Benign or Malignant classifications.

## 🚀 Live Demo

**[Click here to open the Breast Cancer Diagnosis App](https://ibmprojectgit-pzigppusafn4bbv2hnemsq.streamlit.app/)**

## ✨ Features

- Dataset exploration and data cleaning
- Exploratory Data Analysis (EDA) and interactive visualizations
- Feature correlation and importance analysis
- Machine-learning model comparison
- Model evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Interactive Benign / Malignant prediction
- CSV file upload for data exploration

## 📊 Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Records:** 569
- **Features:** 30 numeric features
- **Target:** Diagnosis (Benign or Malignant)

[Dataset Source – Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Jupyter Notebook

## 🤖 Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. Random Forest

## 📁 Project Structure

```text
Breast-Cancer-Diagnosis/
├── app.py
├── utils.py
├── train_models.py
├── data.csv
├── models/
├── requirements.txt
├── README.md
└── Ruthvika_BreastCancerAnalytics.ipynb
```

## ⚙️ Installation and Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download the dataset**

Download `data.csv` from Kaggle and place it in the project folder.

**3. Train the models**

```bash
python train_models.py
```

Skip this step if the trained model files already exist.

**4. Run the application**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 🎓 Acknowledgment

This project was developed as part of the **IBM SkillsBuild Data Analytics with AI Internship Program**, in association with AICTE and BharatCares.

## ⚠️ Disclaimer

This project is intended for educational purposes only. Predictions are not medical diagnoses and must not be used to guide medical decisions. Consult qualified healthcare professionals for medical concerns.

## 👩‍💻 Author

**Ruthvika Nemakal**
