# Updating the code to reflect that the user wants an advanced version
# Including a more feature-rich Streamlit app with enhanced UI and AI chat functionality
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import openai

# Load custom CSS
# Load custom CSS from static folder
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Advanced CSV Analyzer", layout="wide")

# Fix: load CSS from the 'static' subfolder
local_css("static/style.css")
st.markdown("<h1 class='title'>📊 Advanced CSV/Excel Visualizer & AI Assistant</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to read the file: {e}")
        st.stop()

    st.subheader("📌 Data Preview")
    st.dataframe(df)

    st.subheader("🧮 Summary Statistics")
    st.write(df.describe(include='all'))

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    st.subheader("📈 Visualizations")
    chart_type = st.selectbox("Chart type", ['Line Chart', 'Bar Chart', 'Boxplot', 'Scatter Plot', 'Correlation Heatmap'])

    x_col = st.selectbox("X-axis", df.columns)
    y_col = st.selectbox("Y-axis (numeric)", numeric_cols)

    if chart_type == 'Line Chart':
        st.line_chart(df.set_index(x_col)[y_col])
    elif chart_type == 'Bar Chart':
        fig = plt.figure()
        sns.barplot(x=df[x_col], y=df[y_col])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    elif chart_type == 'Boxplot':
        fig = plt.figure()
        sns.boxplot(x=df[x_col], y=df[y_col])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    elif chart_type == 'Scatter Plot':
        fig = plt.figure()
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=categorical_cols[0] if categorical_cols else None)
        st.pyplot(fig)
    elif chart_type == 'Correlation Heatmap':
        fig = plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)

    st.subheader("🔮 Predictive Modeling")
    task = st.radio("Select task", ["Regression", "Classification"])
    target_col = st.selectbox("Target variable", df.columns)
    feature_cols = st.multiselect("Feature columns", [col for col in df.columns if col != target_col])

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        try:
            X = pd.get_dummies(X, drop_first=True)
            if df[target_col].dtype == 'object':
                y = pd.factorize(y)[0]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if task == "Regression":
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = mean_squared_error(y_test, preds, squared=False)
                st.success(f"RMSE: {score:.2f}")
            else:
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.success(f"Accuracy: {acc*100:.2f}%")

            st.write("Sample Predictions:")
            st.write(pd.DataFrame({"Actual": y_test[:5], "Predicted": preds[:5]}))
        except Exception as e:
            st.error(f"Modeling error: {e}")

    st.subheader("💬 Chat with AI about your Data")
    user_question = st.text_input("Ask a question about your data")

    if user_question:
        try:
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're a helpful data assistant."},
                    {"role": "user", "content": f"The dataset has these columns: {df.columns.tolist()}. Here is a sample:\n{df.head().to_string()}\n\nQuestion: {user_question}"}
                ]
            )
            st.write(completion.choices[0].message['content'])
        except Exception as e:
            st.error(f"Chat error: {e}")
