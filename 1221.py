import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
import graphviz
import io
import base64

st.set_page_config(layout="wide", page_title="Customer Dashboard", page_icon="📊")

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .block-container {
        padding-top: 2rem;
    }
    .title-style {
        font-size: 3rem;
        font-weight: bold;
        color: #e74c3c;  /* Red color for title */
        text-align: center;
    }
    summary {
        font-size: 1.1rem;
        font-weight: bold;
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        cursor: pointer;
    }
    details {
        margin-bottom: 10px;
    }
    /* Styling for the tables */
    .red-table th, .red-table td {\n  border: 1px solid #ddd;\n  padding: 8px;\n  background-color: #fdd;\n}\n.red-table th {\n  background-color: #e74c3c;\n  color: white;\n}\n\n.green-table th, .green-table td {\n  border: 1px solid #ddd;\n  padding: 8px;\n  background-color: #dfedda;\n}\n.green-table th {\n  background-color: #27ae60;\n  color: white;\n}\n    /* CSS for hover-expand cluster visualizations */\n    .hover-box {\n        width: 100%;\n        overflow: hidden;\n        transition: transform 0.3s ease;\n        box-shadow: 0 2px 6px rgba(0,0,0,0.1);\n        border-radius: 10px;\n        margin-bottom: 1rem;\n    }\n    .hover-box:hover {\n        transform: scale(1.25);\n        z-index: 10;\n    }\n    .img-wrap {\n        display: flex;\n        justify-content: center;\n    }\n    </style>\n    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    df = pd.read_excel("marketing_campaign1.xlsx")
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
    return df

def preprocess_data(data):
    data['Age'] = 2025 - data['Year_Birth']
    data['Total_Spending'] = data[['MntWines', 'MntFruits', 'MntMeatProducts',
                                   'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    cap_val = data['Total_Spending'].quantile(0.99)
    capped_count = (data['Total_Spending'] > cap_val).sum()
    data['Total_Spending'] = np.where(data['Total_Spending'] > cap_val, cap_val, data['Total_Spending'])
    
    median_income = data['Income'].median()
    data['Income'].fillna(median_income, inplace=True)
    
    Q1 = data['Income'].quantile(0.25)
    Q3 = data['Income'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = len(data[(data['Income'] < lower) | (data['Income'] > upper)])
    data['Income'] = np.clip(data['Income'], lower, upper)
    
    data['Children'] = data['Kidhome'] + data['Teenhome']
    data.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True)
    
    data['Marital_Group'] = data['Marital_Status'].apply(
        lambda x: 'Single' if x in ['Single','YOLO','Absurd','Divorced','Widow','Alone'] else 'Family')
    data['Education'] = data['Education'].replace({
        'Basic': 'Undergraduate',
        '2n Cycle': 'Undergraduate',
        'Graduation': 'Graduate',
        'Master': 'Postgraduate',
        'PhD': 'Postgraduate'})
    
    drop_cols = ['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue']
    data.drop(columns=drop_cols, inplace=True)
    return data, drop_cols, capped_count, outliers

def random_forest(data, features, target):
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    conf = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    return report['accuracy'], conf, auc, fpr, tpr, importance.importances_mean, X.columns.tolist()

def cluster_visuals(data, k_value=2):
    # We'll set k_value for KMeans dynamically
    clustering_cols = ['Income', 'Age', 'Total_Spending']
    scaled = StandardScaler().fit_transform(data[clustering_cols])
    pca = PCA(n_components=2).fit_transform(scaled)
    data['PCA1'], data['PCA2'] = pca[:,0], pca[:,1]
    
    plots = {}
    
    kmeans = KMeans(n_clusters=k_value, random_state=42)
    data['KMeans'] = kmeans.fit_predict(scaled)
    fig1, ax1 = plt.subplots(figsize=(4,3))
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='KMeans', palette='tab10', ax=ax1)
    ax1.set_title(f"KMeans Clustering (k={k_value})")
    plots['KMeans'] = fig1
    data.drop(columns=['KMeans'], inplace=True)
    
    fig2, ax2 = plt.subplots(figsize=(4,3))
    sch.dendrogram(sch.linkage(scaled, method='ward'), no_labels=True, ax=ax2)
    ax2.set_title("Agglomerative Dendrogram (k=2)")
    plots['Agglomerative'] = fig2
    
    gmm = GaussianMixture(n_components=2, random_state=42)
    data['GMM'] = gmm.fit_predict(scaled)
    fig3, ax3 = plt.subplots(figsize=(4,3))
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='GMM', palette='tab10', ax=ax3)
    ax3.set_title("GMM Clustering (k=2)")
    plots['GMM'] = fig3
    data.drop(columns=['GMM'], inplace=True)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    data['DBSCAN'] = dbscan.fit_predict(scaled)
    fig4, ax4 = plt.subplots(figsize=(4,3))
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='DBSCAN', palette='tab10', ax=ax4)
    ax4.set_title("DBSCAN Clustering")
    plots['DBSCAN'] = fig4
    
    return plots

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64

def main():
    st.markdown('<div class="title-style">🚀 Customer Segmentation & Prediction Dashboard 🎯</div>', unsafe_allow_html=True)
    df = load_data()
    df, dropped, cap_count, outlier_count = preprocess_data(df)
    
    # Sidebar Filters
    col_filters = st.sidebar
    col_filters.header("🔎 Filters")
    selected_edu = col_filters.multiselect("🎓 Select Education Level:",
                                           options=df['Education'].unique(),
                                           default=df['Education'].unique())
    selected_marital = col_filters.multiselect("💍 Select Marital Group:",
                                               options=df['Marital_Group'].unique(),
                                               default=df['Marital_Group'].unique())
    income_range = col_filters.slider("💰 Select Income Range:",
                                      min_value=int(df['Income'].min()),
                                      max_value=int(df['Income'].max()),
                                      value=(int(df['Income'].min()), int(df['Income'].max())))
    
    df_filtered = df[(df['Education'].isin(selected_edu)) &
                     (df['Marital_Group'].isin(selected_marital)) &
                     (df['Income'] >= income_range[0]) & (df['Income'] <= income_range[1])]
    
    features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group', 'Children']
    
    # Let user set k for KMeans in sidebar
    k_value = col_filters.slider("Select number of clusters for KMeans", min_value=2, max_value=10, value=2, step=1)
    
    accuracy, conf, auc, fpr, tpr, importances, feature_names = random_forest(df_filtered, features, 'Response')
    plots = cluster_visuals(df_filtered, k_value=k_value)
    
    # Left Column: Data & Model Insights with interactive expanders
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📌 Model & Data Insights")
        st.metric("✅ Accuracy", f"{accuracy*100:.2f}%")
        st.write(f"📦 Capped Spending values: {cap_count}")
        st.write(f"📉 Removed Income outliers: {outlier_count}")
        
        # Display Dropped Columns and Features Used side by side as tables
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🧾 Dropped Columns")
            # Build a HTML table for dropped columns with red styling
            red_table = "<table class='red-table'><tr><th>Dropped Columns</th></tr>"
            for col in dropped:
                red_table += f"<tr><td>{col}</td></tr>"
            red_table += "</table>"
            st.markdown(red_table, unsafe_allow_html=True)
        with col_b:
            st.markdown("### 🧮 Features Used")
            green_table = "<table class='green-table'><tr><th>Features Used</th></tr>"
            for feat in features:
                green_table += f"<tr><td>{feat}</td></tr>"
            green_table += "</table>"
            st.markdown(green_table, unsafe_allow_html=True)
            
        st.markdown("### 🤖 Models Applied")
        with st.expander("🌲 Random Forest (Main Model)"):
            st.write("Used for predicting customer responses. Evaluated by accuracy, ROC-AUC, and feature importance. It acts as the backbone for our decision-making process.")
        with st.expander(f"🎯 KMeans (k={k_value})"):
            st.write("Partitions customers into clusters using spending, age, and income data. Helps in grouping customers for targeted marketing.")
        with st.expander("🧩 Agglomerative Clustering"):
            st.write("Builds a dendrogram to explore natural hierarchies in the data. Useful for discovering subgroup structures.")
        with st.expander("🎲 Gaussian Mixture Model (GMM)"):
            st.write("Provides probabilistic clustering with soft assignments. Useful where clusters may overlap.")
        with st.expander("🌌 DBSCAN"):
            st.write("Identifies clusters based on density, automatically recognizing outliers and noise.")
    
    # Right Column: Visualizations (Random Forest Performance)
    with col2:
        st.header("📈 Visualizations")
        st.subheader("📊 Confusion Matrix")
        fig_conf, ax_conf = plt.subplots(figsize=(6,4))
        sns.heatmap(conf, annot=True, cmap='Blues', fmt='d', ax=ax_conf)
        ax_conf.set_title("Confusion Matrix")
        st.pyplot(fig_conf)
        
        st.subheader("📈 ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(6,4))
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax_roc.plot([0,1],[0,1],'k--')
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)
        
        st.subheader("📊 Feature Importance")
        fig_imp, ax_imp = plt.subplots(figsize=(6,4))
        sorted_idx = np.argsort(importances)[::-1]
        sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx], ax=ax_imp, palette='mako')
        ax_imp.set_title("Feature Importance")
        st.pyplot(fig_imp)
    
    # Cluster Visualizations with hover expand effect
    st.markdown("### 🔍 Cluster Visualizations 🌀")
    # Define HTML/CSS block for hover effect (each image expands on hover)
    st.markdown(
        """
        <style>
        .hover-box {
            width: 100%;
            overflow: hidden;
            transition: transform 0.3s ease;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .hover-box:hover {
            transform: scale(1.25);
            z-index: 10;
        }
        .img-wrap {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    clust_cols = st.columns(len(plots))
    for i, (name, fig) in enumerate(plots.items()):
        img_base64 = fig_to_base64(fig)
        with clust_cols[i]:
            st.markdown(
                f"""
                <div class="hover-box">
                    <div class="img-wrap">
                        <img src="data:image/png;base64,{img_base64}" width="100%">
                    </div>
                    <p style="text-align: center; font-weight: bold;">{name}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
if __name__ == "__main__":
    main()
