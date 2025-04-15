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
import streamlit.components.v1 as components
import base64
from io import BytesIO

st.set_page_config(layout="wide", page_title="Customer Dashboard", page_icon="📊")

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

    data['Marital_Group'] = data['Marital_Status'].apply(lambda x: 'Single' if x in ['Single','YOLO','Absurd','Divorced','Widow','Alone'] else 'Family')
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

def cluster_visuals(data):
    clustering_cols = ['Income', 'Age', 'Total_Spending']
    scaled = StandardScaler().fit_transform(data[clustering_cols])
    pca = PCA(n_components=2).fit_transform(scaled)
    data['PCA1'], data['PCA2'] = pca[:,0], pca[:,1]

    plots = {}

    def fig_to_html(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        html = f"""
        <div style='transition: transform 0.3s ease; margin: 10px; width: 250px; height: auto; overflow: hidden;'>
            <img src='data:image/png;base64,{encoded}' style='width:100%; height:auto; border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.3s ease-in-out;' onmouseover="this.style.transform='scale(1.2)'" onmouseout="this.style.transform='scale(1)'"/>
        </div>
        """
        return html

    kmeans = KMeans(n_clusters=2, random_state=42)
    data['KMeans'] = kmeans.fit_predict(scaled)
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='KMeans', palette='tab10', ax=ax1)
    ax1.set_title("KMeans Clustering (k=2)")
    plots['KMeans'] = fig_to_html(fig1)
    data.drop(columns=['KMeans'], inplace=True)

    fig2, ax2 = plt.subplots()
    sch.dendrogram(sch.linkage(scaled, method='ward'), no_labels=True, ax=ax2)
    ax2.set_title("Agglomerative Dendrogram (k=2)")
    plots['Agglomerative'] = fig_to_html(fig2)

    gmm = GaussianMixture(n_components=2, random_state=42)
    data['GMM'] = gmm.fit_predict(scaled)
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='GMM', palette='tab10', ax=ax3)
    ax3.set_title("GMM Clustering (k=2)")
    plots['GMM'] = fig_to_html(fig3)
    data.drop(columns=['GMM'], inplace=True)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    data['DBSCAN'] = dbscan.fit_predict(scaled)
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='DBSCAN', palette='tab10', ax=ax4)
    ax4.set_title("DBSCAN Clustering")
    plots['DBSCAN'] = fig_to_html(fig4)

    return plots

def main():
    st.title("📊 Customer Segmentation & Prediction Dashboard")
    df = load_data()
    df, dropped, cap_count, outlier_count = preprocess_data(df)

    col_filters = st.sidebar
    col_filters.header("🔎 Filters")
    selected_edu = col_filters.multiselect("🎓 Select Education Level:", options=df['Education'].unique(), default=df['Education'].unique())
    selected_marital = col_filters.multiselect("💍 Select Marital Group:", options=df['Marital_Group'].unique(), default=df['Marital_Group'].unique())
    income_range = col_filters.slider("💰 Select Income Range:", min_value=int(df['Income'].min()), max_value=int(df['Income'].max()), value=(int(df['Income'].min()), int(df['Income'].max())))

    df_filtered = df[(df['Education'].isin(selected_edu)) &
                     (df['Marital_Group'].isin(selected_marital)) &
                     (df['Income'] >= income_range[0]) & (df['Income'] <= income_range[1])]

    features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group', 'Children']
    accuracy, conf, auc, fpr, tpr, importances, feature_names = random_forest(df_filtered, features, 'Response')
    plots = cluster_visuals(df_filtered)

    st.header("📌 Model Insights")
    st.metric("✅ Accuracy", f"{accuracy*100:.2f}%")

    st.subheader("🌀 Cluster Visualizations")
    html_content = "".join(plots.values())
    components.html(f"""
        <div style='display:flex; flex-wrap: wrap; justify-content: center;'>
            {html_content}
        </div>
    """, height=600, scrolling=True)

if __name__ == '__main__':
    main()
