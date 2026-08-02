#!/usr/bin/env python
# coding: utf-8

# ## Customer Segmentation & Churn Prediction — Streamlit Web App
# 
# **Important:**  
# Run the notebook `notebooks/capstone_project.ipynb` **at least once** before launching the app.  
# This is because the app depends on artifacts generated in `/reports`:
# 
# - `scored_customers.csv`  
# - `churn_model.pkl`  
# - `kmeans_model.pkl`  
# - `scaler.pkl`  
# - `feature_columns.pkl`  
# - `input_schema.json`  
# - `model_metrics.csv`  
# - `feature_importance.csv`  
# - `cluster_profile.csv`
# 
# ---
# 
# ### Launch Instructions
# 
# 1. Navigate to the dashboard folder:
#    ```bash
#    cd dashboard
# 

# In[ ]:


import json
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


# In[ ]:


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & Segmentation",
    page_icon="📊",
    layout="wide",
)
REPORTS_DIR = "../reports"
REQUIRED_FILES = [
    "scored_customers.csv", "churn_model.pkl", "kmeans_model.pkl", "scaler.pkl",
    "feature_columns.pkl", "input_schema.json", "model_metrics.csv",
    "feature_importance.csv", "cluster_profile.csv",
]
missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(REPORTS_DIR, f))]

if missing:
    st.error(
        "Missing required files: " + ", ".join(missing) +
        "\n\nRun `notebooks/capstone_project.ipynb` all the way through first — "
        "it generates everything this app needs inside `/reports`."
    )
    st.stop()


# In[ ]:


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(REPORTS_DIR, "scored_customers.csv"))


@st.cache_resource
def load_models():
    churn_model = joblib.load(os.path.join(REPORTS_DIR, "churn_model.pkl"))
    kmeans_model = joblib.load(os.path.join(REPORTS_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(REPORTS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(REPORTS_DIR, "feature_columns.pkl"))
    return churn_model, kmeans_model, scaler, feature_columns


@st.cache_data
def load_schema():
    with open(os.path.join(REPORTS_DIR, "input_schema.json")) as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    metrics = pd.read_csv(os.path.join(REPORTS_DIR, "model_metrics.csv"))
    importance = pd.read_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index_col=0)
    cluster_profile = pd.read_csv(os.path.join(REPORTS_DIR, "cluster_profile.csv"), index_col=0)
    return metrics, importance, cluster_profile


df = load_data()
churn_model, kmeans_model, scaler, feature_columns = load_models()
schema = load_schema()
model_metrics, feature_importance, cluster_profile = load_metrics()

CAT_OPTIONS = schema["categorical_options"]
NUM_RANGES = schema["numeric_ranges"]


# In[ ]:


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Exploratory Analysis", "Customer Segments", "Predict Churn", "Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "ML Capstone Project — combines K-Means clustering (unsupervised) with a "
    "tuned classifier (supervised) to identify at-risk customer segments."
)


# In[ ]:


# ==================================================================
# PAGE 1: OVERVIEW
# ==================================================================
if page == "Overview":
    st.title("Customer Segmentation & Churn Prediction")
    st.markdown(
        "This project segments telecom customers into behavioral groups using "
        "**unsupervised learning**, and predicts churn risk using **supervised learning** — "
        "then combines both to highlight which segments need retention attention."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Overall Churn Rate", f"{df['Churn'].mean() * 100:.1f}%")
    c3.metric("Customer Segments", df["Cluster"].nunique())
    c4.metric("High-Risk Customers (>70%)", f"{(df['Churn_Probability'] > 0.7).sum():,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
        ax.pie(
            churn_counts.values, labels=churn_counts.index, autopct="%1.1f%%",
            colors=["#2E86AB", "#E63946"], startangle=90,
        )
        st.pyplot(fig)

    with col_b:
        st.subheader("Churn Rate by Segment")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.barplot(x=cluster_profile.index, y=cluster_profile["Churn_Rate_%"], palette="rocket", ax=ax2)
        ax2.set_xlabel("Segment")
        ax2.set_ylabel("Churn Rate (%)")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Business Recommendations")
    riskiest = cluster_profile["Churn_Rate_%"].idxmax()
    safest = cluster_profile["Churn_Rate_%"].idxmin()
    st.markdown(
        f"- **Segment {riskiest}** has the highest churn rate "
        f"({cluster_profile.loc[riskiest, 'Churn_Rate_%']:.1f}%) — prioritize retention offers here.\n"
        f"- **Segment {safest}** is the most loyal ({cluster_profile.loc[safest, 'Churn_Rate_%']:.1f}% churn) "
        f"— low priority for retention spend.\n"
        f"- Use the **Predict Churn** page to score a specific customer, or **Model Performance** "
        f"to see which factors drive churn most."
    )


# In[ ]:


# ==================================================================
# PAGE 2: EDA
# ==================================================================
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tenure Distribution by Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=ax, palette=["#2E86AB", "#E63946"])
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Charges by Churn")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30, kde=True, ax=ax2, palette=["#2E86AB", "#E63946"])
        st.pyplot(fig2)

    st.markdown("---")
    if "Contract" in df.columns:
        st.subheader("Churn Rate by Contract Type")
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=df, x="Contract", y="Churn",
            estimator=lambda x: sum(x) / len(x) * 100, palette="viridis", ax=ax3,
        )
        ax3.set_ylabel("Churn Rate (%)")
        st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(50), use_container_width=True)


# In[ ]:


# ==================================================================
# PAGE 3: SEGMENTS
# ==================================================================
elif page == "Customer Segments":
    st.title("Customer Segments (Unsupervised Learning)")
    st.markdown(
        "Segments were created with **K-Means clustering** on tenure, monthly charges, "
        "and total charges — without using the churn label."
    )

    st.subheader("Segment Profiles")
    st.dataframe(cluster_profile, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df, x="tenure", y="MonthlyCharges", hue="Cluster",
            palette="viridis", alpha=0.6, ax=ax,
        )
        ax.set_title("Segments: Tenure vs Monthly Charges")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.barplot(x=cluster_profile.index, y=cluster_profile["Count"], palette="mako", ax=ax2)
        ax2.set_xlabel("Segment")
        ax2.set_ylabel("Number of Customers")
        ax2.set_title("Segment Sizes")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Explore a Segment")
    chosen_cluster = st.selectbox("Select a segment", sorted(df["Cluster"].unique()))
    seg_df = df[df["Cluster"] == chosen_cluster]
    st.write(f"**{len(seg_df)} customers** in Segment {chosen_cluster} — "
             f"**{seg_df['Churn'].mean() * 100:.1f}% churn rate**")
    st.dataframe(
        seg_df[["tenure", "MonthlyCharges", "TotalCharges", "Churn", "Churn_Probability"]].head(20),
        use_container_width=True,
    )


# In[ ]:


# ==================================================================
# PAGE 4: PREDICT CHURN (interactive form)
# ==================================================================
elif page == "Predict Churn":
    st.title("Predict Churn for a Customer")
    st.markdown("Fill in customer details below to get a live churn risk prediction from the trained model.")

    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", CAT_OPTIONS["gender"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", CAT_OPTIONS["Partner"])
            dependents = st.selectbox("Has Dependents", CAT_OPTIONS["Dependents"])

        with col2:
            st.markdown("**Account Info**")
            tenure = st.slider("Tenure (months)", 0, int(NUM_RANGES["tenure"]["max"]), 12)
            contract = st.selectbox("Contract", CAT_OPTIONS["Contract"])
            paperless = st.selectbox("Paperless Billing", CAT_OPTIONS["PaperlessBilling"])
            payment = st.selectbox("Payment Method", CAT_OPTIONS["PaymentMethod"])
            monthly_charges = st.slider(
                "Monthly Charges ($)",
                float(NUM_RANGES["MonthlyCharges"]["min"]),
                float(NUM_RANGES["MonthlyCharges"]["max"]),
                float(NUM_RANGES["MonthlyCharges"]["mean"]),
            )

        with col3:
            st.markdown("**Services**")
            phone = st.selectbox("Phone Service", CAT_OPTIONS["PhoneService"])
            multiple_lines = st.selectbox("Multiple Lines", CAT_OPTIONS["MultipleLines"])
            internet = st.selectbox("Internet Service", CAT_OPTIONS["InternetService"])
            online_security = st.selectbox("Online Security", CAT_OPTIONS["OnlineSecurity"])
            online_backup = st.selectbox("Online Backup", CAT_OPTIONS["OnlineBackup"])
            device_protection = st.selectbox("Device Protection", CAT_OPTIONS["DeviceProtection"])
            tech_support = st.selectbox("Tech Support", CAT_OPTIONS["TechSupport"])
            streaming_tv = st.selectbox("Streaming TV", CAT_OPTIONS["StreamingTV"])
            streaming_movies = st.selectbox("Streaming Movies", CAT_OPTIONS["StreamingMovies"])

        total_charges = st.number_input(
            "Total Charges ($) — leave as tenure × monthly if unsure",
            value=round(tenure * monthly_charges, 2),
        )

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        raw_input = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])

        # One-hot encode exactly like training, then align columns
        encoded_input = pd.get_dummies(raw_input)
        encoded_input = encoded_input.reindex(columns=feature_columns, fill_value=0)

        churn_prob = churn_model.predict_proba(encoded_input)[0, 1]
        churn_pred = "Likely to Churn" if churn_prob >= 0.5 else "Likely to Stay"

        # Also assign this customer to a segment
        cluster_input = pd.DataFrame(
            [[tenure, monthly_charges, total_charges]],
            columns=["tenure", "MonthlyCharges", "TotalCharges"],
        )
        cluster_features_scaled = scaler.transform(cluster_input)
        segment = kmeans_model.predict(cluster_features_scaled)[0]

        st.markdown("---")
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("Churn Probability", f"{churn_prob * 100:.1f}%")
        rcol2.metric("Prediction", churn_pred)
        rcol3.metric("Assigned Segment", f"Segment {segment}")

        if churn_prob >= 0.5:
            st.warning(
                "⚠️ This customer is at elevated risk of churning. Consider a retention offer, "
                "contract upgrade incentive, or loyalty discount."
            )
        else:
            st.success("✅ This customer looks likely to stay based on current profile.")


# In[ ]:


# ==================================================================
# PAGE 5: MODEL PERFORMANCE
# ==================================================================
elif page == "Model Performance":
    st.title("Model Performance (Supervised Learning)")

    st.subheader("Model Comparison")
    st.dataframe(model_metrics, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    model_metrics_sorted = model_metrics.sort_values("F1-Score", ascending=True)
    ax.barh(model_metrics_sorted["Model"], model_metrics_sorted["F1-Score"], color="#2E86AB")
    ax.set_xlabel("F1-Score")
    ax.set_title("Model Comparison by F1-Score")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top Churn-Driving Features")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    top_features = feature_importance.sort_values("importance", ascending=True).tail(15)
    ax2.barh(top_features.index, top_features["importance"], color="#E63946")
    ax2.set_xlabel("Importance")
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Highest Churn-Risk Customers")
    top_n = st.slider("Show top N highest-risk customers", 5, 50, 15)
    risky = df.sort_values("Churn_Probability", ascending=False).head(top_n)
    st.dataframe(
        risky[["tenure", "MonthlyCharges", "TotalCharges", "Cluster", "Churn_Probability"]],
        use_container_width=True,
    )


# In[ ]:




