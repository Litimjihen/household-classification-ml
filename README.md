# household-classification-ml
Machine Learning project: predicting number of children using socio-demographic data (8,400 observations)
# 🏠 Household Classification - Predicting Number of Children

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

Machine learning classification project to predict the number of children in households based on **8,400 socio-demographic observations**.

This project demonstrates end-to-end data science workflow from exploratory data analysis to model deployment.

## 🎯 Business Objective

Build and compare classification models to identify key factors influencing family size decisions, providing actionable insights for:
- Social policy targeting
- Demographic forecasting
- Resource allocation planning

## 📊 Dataset

- **Size**: 8,400 observations
- **Features**: Socio-demographic variables including:
  - Income level
  - Education
  - Occupation
  - Geographic location
  - Age
  - Marital status
- **Target Variable**: Number of children (multi-class classification)

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Libraries**: 
  - Data Processing: Pandas, NumPy
  - Machine Learning: Scikit-learn
  - Visualization: Matplotlib, Seaborn
  - Statistical Testing: SciPy
- **Environment**: Jupyter Notebook

## 🔍 Methodology

### 1. Data Preprocessing
- Missing values imputation
- Categorical variables encoding (One-Hot, Label Encoding)
- Feature scaling and normalization
- Train-test split (80/20)

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics
- Distribution analysis
- Correlation heatmaps
- Statistical hypothesis testing:
  - Chi-square test
  - ANOVA
  - MANOVA
  - Normality tests

### 3. Feature Engineering
- Variable selection based on correlation and domain knowledge
- Feature importance analysis
- Dimensionality reduction exploration (PCA)

### 4. Model Training & Evaluation

**Models Tested:**
- Logistic Regression (baseline)
- Decision Tree
- Random Forest (best performer)

**Evaluation Metrics:**
- ✅ **Accuracy**: 75%
- ✅ **F1-Score**: 0.72
- Confusion Matrix
- Precision & Recall per class

## 📈 Key Results

### Model Performance Comparison

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Logistic Regression | 68% | 0.65 | 2s |
| Decision Tree | 71% | 0.68 | 5s |
| **Random Forest** | **75%** | **0.72** | 15s |

### Top 3 Predictive Features
1. **Income Level** (importance: 0.35)
2. **Education** (importance: 0.28)
3. **Age** (importance: 0.22)

### Insights
- Higher education correlates with fewer children
- Income shows non-linear relationship with family size
- Geographic location is less predictive than expected

## 🚀 How to Run

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/Litimjihen/household-classification-ml.git
cd household-classification-ml

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis
```bash
# Launch Jupyter Notebook
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_EDA.ipynb
# 2. notebooks/02_preprocessing.ipynb
# 3. notebooks/03_modeling.ipynb
```

## 📁 Project Structure
```
household-classification-ml/
│
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned data
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data cleaning & feature engineering
│   └── 03_modeling.ipynb      # Model training & evaluation
│
├── src/
│   ├── data_preprocessing.py  # Data cleaning functions
│   ├── feature_engineering.py # Feature creation utilities
│   └── model_training.py      # Model training pipeline
│
├── results/
│   ├── figures/          # Visualizations
│   └── models/           # Saved models (.pkl)
│
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Git ignore rules
└── LICENSE              # MIT License
```

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] Test ensemble methods (XGBoost, LightGBM, CatBoost)
- [ ] Implement SMOTE for class imbalance handling
- [ ] Deploy model as REST API (FastAPI)
- [ ] Add SHAP values for model interpretability
- [ ] Create interactive dashboard with Streamlit

## 📚 Learnings

This project reinforced my understanding of:
- Complete ML workflow from raw data to production-ready model
- Statistical testing for data validation
- Feature engineering impact on model performance
- Importance of model interpretability in business contexts

## 👤 Author

**Jihen LITIM**  
MSc Artificial Intelligence Student | Former Embedded Systems Engineer  

📧 [jihen.litim@aivancity.education](mailto:jihen.litim@aivancity.education)  
🔗 [LinkedIn](https://www.linkedin.com/in/votre-profil)  
🐙 [GitHub](https://github.com/Litimjihen)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project helpful, please consider giving it a star!**
