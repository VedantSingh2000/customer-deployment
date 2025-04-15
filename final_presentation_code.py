import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# --- Set page configuration ---
st.set_page_config(layout="wide")

# --- Custom CSS for hover effects and styling ---
st.markdown("""
<style>
.graph-container {
    width: 180px;
    height: 140px;
    transition: all 0.3s ease-in-out;
    overflow: hidden;
    margin: 10px;
    border: 1px solid #ddd;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.graph-container:hover {
    width: 400px;
    height: 300px;
    z-index: 10;
}
.title-highlight {
    font-size: 14px;
    font-weight: bold;
    color: #3b82f6;
    text-align: center;
}
.metric-label {
    font-weight: bold;
    color: #16a34a;
}
.sidebar-header {
    font-size: 22px;
    font-weight: bold;
    color: #2563eb;
    margin-bottom: 10px;
}
.highlighted-model {
    border: 2px solid #ff4b4b !important; /* Highlight color for Random Forest */
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    return df

def feature_engineering(df_raw):
    df = df_raw.copy()
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
    median_income = df['Income'].median()
    df['Income'] = df['Income'].fillna(median_income)
    Q1 = df['Income'].quantile(0.25)
    Q3 = df['Income'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    income_outliers_count = ((df['Income'] < lower_bound) | (df['Income'] > upper_bound)).sum()
    df['Income'] = np.clip(df['Income'], lower_bound, upper_bound)
    df['Age'] = 2025 - df['Year_Birth']
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spending'] = df[spending_cols].sum(axis=1)
    spending_cap = df['Total_Spending'].quantile(0.99)
    spending_capped_count = (df['Total_Spending'] > spending_cap).sum()
    df['Total_Spending'] = np.where(df['Total_Spending'] > spending_cap, spending_cap, df['Total_Spending'])
    df['Children'] = df['Kidhome'] + df['Teenhome']
    df['Marital_Group'] = df['Marital_Status'].apply(lambda x: 'Single' if x in ['Single', 'Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'] else 'Family')
    df['Education'] = df['Education'].replace({'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate', 'Graduation': 'Graduate', 'Master': 'Postgraduate', 'PhD': 'Postgraduate'})
    dropped_cols = ['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Marital_Status', 'Kidhome', 'Teenhome']
    df.drop(dropped_cols, axis=1, inplace=True)
    return df, dropped_cols, spending_capped_count, income_outliers_count

def train_rf(data, features, target='Response'):
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = classification_report(y_test, model.predict(X_test), output_dict=True)['accuracy']
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    roc_fig, roc_ax = plt.subplots(figsize=(4, 3))
    roc_ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    roc_ax.plot([0, 1], [0, 1], 'k--')
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curve")
    roc_ax.legend(loc="lower right")
    importances = model.feature_importances_
    feature_names = list(X.columns)
    feature_importance_fig, feature_importance_ax = plt.subplots(figsize=(4, 3))
    sns.barplot(x=importances, y=feature_names, ax=feature_importance_ax)
    feature_importance_ax.set_title("Feature Importance")
    return accuracy, roc_fig, feature_importance_fig

def clustering_graphs(data):
    cluster_features = ['Income', 'Age', 'Total_Spending']
    X = StandardScaler().fit_transform(data[cluster_features])
    X_pca = PCA(n_components=2).fit_transform(X)
    data['PCA1'], data['PCA2'] = X_pca[:, 0], X_pca[:, 1]
    figs = {}
    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    data['KMeans_Cluster'] = kmeans.fit_predict(X)
    fig_kmeans, ax_kmeans = plt.subplots(figsize=(4, 3))
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='KMeans_Cluster', palette='viridis', ax=ax_kmeans)
    ax_kmeans.set_title("KMeans (k=2)")
    figs['KMeans'] = fig_kmeans
    data.drop('KMeans_Cluster', axis=1, inplace=True)
    # Agglomerative
    agglo = AgglomerativeClustering(n_clusters=2)
    data['Agglo_Cluster'] = agglo.fit_predict(X)
    fig_agglo, ax_agglo = plt.subplots(figsize=(4, 3))
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Agglo_Cluster', palette='plasma', ax=ax_agglo)
    ax_agglo.set_title("Agglomerative (k=2)")
    figs['Agglomerative'] = fig_agglo
    data.drop('Agglo_Cluster', axis=1, inplace=True)
    # GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    data['GMM_Cluster'] = gmm.fit_predict(X)
    fig_gmm, ax_gmm = plt.subplots(figsize=(4, 3))
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='GMM_Cluster', palette='coolwarm', ax=ax_gmm)
    ax_gmm.set_title("GMM (k=2)")
    figs['GMM'] = fig_gmm
    data.drop('GMM_Cluster', axis=1, inplace=True)
    return figs

def main():
    st.title("🧠 Customer Segmentation Dashboard")

    # --- Load and transform data ---
    df_raw = load_data("marketing_campaign1.xlsx")
    df, dropped_cols, spending_capped_count, income_outliers_count = feature_engineering(df_raw.copy())

    # --- Sidebar Filter Options ---
    st.sidebar.markdown("<div class='sidebar-header'>Filter Options</div>", unsafe_allow_html=True)

    edu_options = list(df["Education"].unique())
    selected_edu = st.sidebar.multiselect("Select Education Level", options=edu_options, default=edu_options)

    income_range = st.sidebar.slider("Income Range", int(df["Income"].min()), int(df["Income"].max()), (int(df["Income"].min()), int(df["Income"].max())))

    rel_options = list(df["Marital_Group"].unique())
    selected_rel = st.sidebar.multiselect("Select Relationship (Single/Family)", options=rel_options, default=rel_options)

    # --- Apply filters to data ---
    filtered_df = df[
        (df["Education"].isin(selected_edu)) &
        (df["Income"] >= income_range[0]) & (df["Income"] <= income_range[1]) &
        (df["Marital_Group"].isin(selected_rel))
    ].copy() # Use a copy to avoid SettingWithCopyWarning

    # --- Display Model Information ---
    st.header("🛠️ Models Used")
    st.write("This application employs the following models for customer segmentation and analysis:")
    st.markdown("- **<span class='highlighted-model'>Random Forest</span>:** Used for predicting customer response and feature importance analysis.", unsafe_allow_html=True)
    st.markdown("- **K-Means Clustering (k=2):** Partitions customers into distinct groups based on their features.")
    st.markdown("- **Agglomerative Clustering (k=2):** A hierarchical clustering method to group customers.")
    st.markdown("- **Gaussian Mixture Model (GMM, k=2):** A probabilistic model for clustering customers.")

    st.subheader("Random Forest Performance")
    used_features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group', 'Children']
    accuracy, roc_fig, feature_importance_fig = train_rf(filtered_df.copy(), used_features)
    st.metric("Random Forest Accuracy", f"{accuracy:.2%}")

    st.subheader("Preprocessing Details")
    st.write(f"**Columns Removed:** {', '.join(dropped_cols)}")
    st.write(f"**Number of Income Outliers Capped:** {income_outliers_count}")
    st.write(f"**Number of Total Spending Values Capped (at 99th percentile):** {spending_capped_count}")
    st.write("**Feature Engineering Performed:**")
    st.markdown("- Calculated **Age** from 'Year_Birth'.")
    st.markdown("- Calculated **Total Spending** by summing monetary columns.")
    st.markdown("- Capped extreme values in **Total Spending**.")
    st.markdown("- Imputed missing **Income** values with the median.")
    st.markdown("- Clipped **Income** outliers using the IQR method.")
    st.markdown("- Combined 'Kidhome' and 'Teenhome' into **Children**.")
    st.markdown("- Grouped 'Marital_Status' into **Marital_Group** ('Single' and 'Family').")
    st.markdown("- Simplified **Education** levels.")

    st.divider()

    # --- Random Forest Highlights with Checkbox ---
    st.header("⭐ Random Forest Model Highlights")
    show_rf_graphs = st.checkbox("Show Random Forest Graphs (ROC Curve & Feature Importance)")

    if show_rf_graphs:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.pyplot(roc_fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.subheader("Feature Importance")
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.pyplot(feature_importance_fig)
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # --- Clustering Visualizations ---
    st.header("🌀 Clustering Model Visualizations (k=2)")
    st.markdown("Hover over each scatter plot to increase its size.")
    cluster_figs = clustering_graphs(filtered_df.copy())

    cols = st.columns(3)
    model_names = ["KMeans", "Agglomerative", "GMM"]

    for i, name in enumerate(model_names):
        if name in cluster_figs:
            with cols[i % 3]:
                st.markdown(f"<div class='title-highlight'>{name}</div>", unsafe_allow_html=True)
                st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
                st.pyplot(cluster_figs[name])
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"Clustering plot for {name} not found.")

if __name__ == "__main__":
    main()
