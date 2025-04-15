# 🧠 Customer Personality Analysis Dashboard

A dynamic dashboard built using **Streamlit** to analyze customer segments and evaluate marketing strategies based on behavioral data. This project helps businesses **identify potential customers** and design **targeted marketing campaigns**.

---

## 📊 Dashboard Features

### 🔍 Main Components:
- ✅ **Random Forest Model Accuracy**
- 📉 **ROC Curve** and 📊 **Feature Importance Graph**
- 🌀 **K-Means, Agglomerative, DBSCAN, GMM Clustering** (k=2 by default)
- 🔍 **Interactive Cluster Visuals** (expand on hover)
- 📌 **Insights and Model Performance Summary**

### 🧩 Sidebar Filters:
- 💍 **Marital Group** (`Family`, `Single`)
- 🎓 **Education Level** (`Undergraduate`, `Graduate`, `Postgraduate`)
- 💰 **Income Range** (slider)
- 🎯 **Number of Clusters** (for all 4 clustering models)

---

## 📁 Project Structure

. ├── app.py # Streamlit dashboard code ├── models/ │ └── random_forest.pkl # Trained classification model (optional) ├── data/ │ └── marketing_campaign1.xlsx # Original dataset ├── requirements.txt # List of Python dependencies └── README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-personality-analysis.git
cd customer-personality-analysis
2. Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Launch the Streamlit App
bash
Copy
Edit
streamlit run app.py
📌 Data Highlights
Cleaned and Processed Dataset

Created new features like Total_Spending and Age

Capped high spending values at 99th percentile

Removed income outliers using IQR method

Dropped redundant columns: ID, Year_Birth, Dt_Customer, etc.

📈 Model Insights
Random Forest Classifier:

Shows Accuracy, Confusion Matrix, ROC Curve, and Feature Importance

Clustering Models:

KMeans, Agglomerative, DBSCAN, GMM – each visualized on PCA-transformed data

Insights Panel:

Highlights capped values, removed outliers, dropped columns, and model features

🛠️ Tech Stack
Python 3.9+

Streamlit

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Graphviz (optional)

💡 Future Enhancements
Predictive analytics for product recommendations

Cluster profiling for each customer segment

Multi-language dashboard support

Dark mode / theme toggling

👤 Author
Vedant
