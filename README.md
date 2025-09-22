# Heart Disease Prediction Project

An end-to-end machine-learning pipeline to predict heart disease presence using the UCI Heart dataset.  
The project demonstrates **data preprocessing**, **dimensionality reduction**, **feature selection**, **supervised & unsupervised learning**, **hyperparameter tuning**, and a **Streamlit web app** for deployment.

---

## 📂 Project Structure

```
Heart_Disease_Project/
│── data/
│   └── heart_disease.csv                # Original dataset
│── notebooks/
│   ├── 01_data_preprocessing.ipynb      # Cleaning & preprocessing
│   ├── 02_pca_analysis.ipynb            # PCA dimensionality reduction
│   ├── 03_feature_selection.ipynb       # Feature importance, RFE, Chi²
│   ├── 04_supervised_learning.ipynb     # Logistic Regression, SVM, etc.
│   ├── 05_unsupervised_learning.ipynb   # K-Means, Hierarchical clustering
│   ├── 06_hyperparameter_tuning.ipynb   # GridSearchCV & RandomizedSearchCV
│── models/
│   └── final_model.pkl                  # Best Random Forest model
│── ui/
│   └── app.py                           # Streamlit UI for predictions
│── deployment/
│   └── ngrok_setup.txt                  # Steps to expose the app externally
│── results/
│   └── evaluation_metrics.txt           # Model performance summary
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 🚀 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Heart_Disease_Project.git
   cd Heart_Disease_Project
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Reproduce the Workflow

* Explore each Jupyter notebook in the `notebooks/` folder in numeric order.
* Key steps include:
  * **Data preprocessing**: Missing value handling, scaling.
  * **PCA**: Dimensionality reduction and explained-variance visualization.
  * **Feature selection**: Random Forest importance, RFE, Chi-Square.
  * **Model training**: Logistic Regression, Decision Tree, Random Forest, SVM.
  * **Evaluation**: Accuracy, Precision, Recall, F1, ROC/AUC.
  * **Clustering**: K-Means and Hierarchical clustering.
  * **Hyperparameter tuning**: GridSearchCV & RandomizedSearchCV.

---

## 💻 Streamlit App

Run the prediction web app locally:
```bash
streamlit run ui/app.py
```

---

## 🌐 Optional: Public Deployment with Ngrok
Follow the steps in `deployment/ngrok_setup.txt` to expose the Streamlit app to the internet.

---

## 📊 Results

* **Best Model:** Random Forest after hyperparameter tuning.
* **Key Metrics:** See `results/evaluation_metrics.txt` for Accuracy, Precision, Recall, F1, and AUC.

---

## ⚙️ Requirements
Main libraries:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
ucimlrepo
joblib
```
See the full list in `requirements.txt`.

---

## 👤 Author
Mazen Mahmoud  
[GitHub Profile](https://github.com/Mazen-Mahmoud-dev)
