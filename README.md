🧠 Customer Personality Analysis Dashboard
A dynamic dashboard built using Streamlit to analyze customer segments and evaluate marketing strategies based on behavioral data. This project helps businesses identify potential customers and design targeted marketing campaigns.

📊 Dashboard Features
🔍 Main Components:
Random Forest Model Accuracy

ROC Curve and Feature Importance Graph

K-Means, Agglomerative, DBSCAN, GMM Clustering (K=2)

Dynamic Clustering Visuals (expand on hover)

Insights and Model Performance Summary

🧩 Filters:
Marital Status Group (Alone, In Couple)

Education Level (Graduation, Master, PhD, Basic, 2n Cycle)

Income Range (slider)

📁 Project Structure
bash
Copy
Edit
.
├── app.py                  # Streamlit dashboard code
├── models/
│   └── random_forest.pkl   # Trained classification model
├── data/
│   └── cleaned_data.csv    # Cleaned and feature-engineered dataset
├── requirements.txt        # List of Python dependencies
└── README.md               # Project documentation
🚀 How to Run the Project
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/customer-personality-analysis.git
cd customer-personality-analysis
2. Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
Data Overview: Includes customer demographics, purchase behavior, and campaign response.

Feature Engineering:

Total_Spending column created

Outliers removed from income

Dropped irrelevant or sparse columns

Cluster Analysis: Summarized using 4 unsupervised models for better customer understanding.

📈 Model Insights
Classification: Random Forest with accuracy and ROC AUC score

Clustering: Helps identify distinct customer personas

Filters: Dynamically refine visuals based on marital status, education, and income

🛠️ Tech Stack
Python 3.9+

Streamlit

Pandas, Numpy, Scikit-learn

Matplotlib, Seaborn, Plotly

Joblib

💡 Future Enhancements
Add predictive analytics for future purchases

Integrate more interactive visuals

Support for multi-language dashboard

👤 Author
Vedant
📫 Reach out: LinkedIn | Email

