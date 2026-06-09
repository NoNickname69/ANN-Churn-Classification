# 🏦 Customer Churn Prediction — ANN Classifier

A deep learning web app that predicts whether a bank customer is likely to churn, built with an Artificial Neural Network (ANN) and deployed on Streamlit.

🔗 **Live Demo:** [ann-churn-classification.streamlit.app](https://ann-churn-classification-mmsytkebpgwezdbvbl6mvu.streamlit.app/)

---

## 📌 Project Overview

Customer churn is one of the biggest challenges in the banking industry. This project uses an ANN to classify whether a customer will leave the bank based on their profile — helping businesses take proactive retention measures.

- **Model:** Artificial Neural Network (ANN)
- **Dataset:** Bank Customer Churn dataset (10,000 records)
- **Task:** Binary classification (Churn / No Churn)
- **Deployment:** Streamlit web app

---

## 🗂️ Project Structure

```
├── trials.ipynb                   # Exploratory data analysis & model training
├── prediction.ipynb               # Inference testing notebook
├── salary.ipynb                   # Salary regression experiments
├── app.py                         # Streamlit web app
├── model_v2.h5                    # Saved trained ANN model
├── Label_encoder_gender.pkl       # Label encoder for Gender
├── ohe_geo.pkl                    # One-Hot encoder for Geography
├── scaler.pkl                     # Standard scaler for feature scaling
└── README.md
```

---

## 🧠 Model & Features

The model takes 10 customer features as input:

| Feature | Type | Preprocessing |
|---------|------|---------------|
| Credit Score | Numeric | StandardScaler |
| Geography | Categorical | OneHotEncoder (France / Germany / Spain) |
| Gender | Categorical | LabelEncoder (Male / Female) |
| Age | Numeric | StandardScaler |
| Tenure | Numeric | StandardScaler |
| Balance | Numeric | StandardScaler |
| Number of Products | Numeric | StandardScaler |
| Has Credit Card | Binary | — |
| Is Active Member | Binary | — |
| Estimated Salary | Numeric | StandardScaler |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ann-churn-classification.git
cd ann-churn-classification
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
tensorflow==2.15.0
streamlit
scikit-learn
pandas
numpy
```

---

## 🚀 Running the App Locally

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## 🖥️ How the App Works

1. User fills in customer details via sliders and dropdowns
2. Geography is encoded using the saved OneHotEncoder
3. Gender is encoded using the saved LabelEncoder
4. All features are scaled using the saved StandardScaler
5. The ANN predicts a churn probability between 0 and 1
6. If probability > 0.5 → **Likely to churn**, else → **Not likely to churn**

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `trials.ipynb` | Data preprocessing, feature engineering, model training and evaluation |
| `prediction.ipynb` | Loads saved model and artifacts, runs sample predictions |
| `salary.ipynb` | Regression experiments on salary estimation |

---

## 📈 Sample Prediction

| Input | Value |
|-------|-------|
| Credit Score | 600 |
| Geography | France |
| Gender | Male |
| Age | 40 |
| Tenure | 3 |
| Balance | 60,000 |
| Products | 2 |
| Active Member | Yes |
| Estimated Salary | 50,000 |

**Result:** Not likely to churn (probability: 0.03)