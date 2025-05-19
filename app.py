import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html

st.title("📊 Auto EDA Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File successfully uploaded!")

    # Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Statistik dasar
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Pilih kolom untuk plot
    st.subheader("Quick Visualizations")
    x_axis = st.selectbox("X-axis", df.columns)
    y_axis = st.selectbox("Y-axis", df.columns)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax2)
    st.pyplot(fig2)

    # Optional: Profiling otomatis (heavy)
    if st.checkbox("Run Full EDA Report (Pandas Profiling)"):
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        profile_html = profile.to_html()
        html(profile_html, height=1000, scrolling=True)
