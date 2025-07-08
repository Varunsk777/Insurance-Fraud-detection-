import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


#all are checkbox features other than dataset visualization
st.title("Fraud Detection Dashboard")
st.markdown("This dashboard allows you to visualize and preprocess data for fraud detection models.")

# Upload Datasets
st.sidebar.header("Upload Datasets")
uploaded_file_1 = st.sidebar.file_uploader("Upload Dataset 1 (CSV)", type="csv", key="file1")
uploaded_file_2 = st.sidebar.file_uploader("Upload Dataset 2 (CSV)", type="csv", key="file2")

if uploaded_file_1 and uploaded_file_2:
    # Load Datasets
    data1 = pd.read_csv(uploaded_file_1)
    data2 = pd.read_csv(uploaded_file_2)

    # Create two columns for parallel display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset 1 Overview")
        st.dataframe(data1.head())
    with col2:
        st.subheader("Dataset 2 Overview")
        st.dataframe(data2.head())

    if st.checkbox("Check Missing Values in Both Datasets"):#checking for missing values
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Missing Values in Dataset 1:**")
            st.write(data1.isnull().sum())
        with col2:
            st.write("**Missing Values in Dataset 2:**")
            st.write(data2.isnull().sum())

    # Preprocessing Section
    def preprocess_data(data, dataset_name):
        st.subheader(f"Data Preprocessing - {dataset_name}")
        if st.checkbox(f"Drop Missing Values in {dataset_name}"):#drop missing values
            data.dropna(inplace=True)
            st.success(f"Missing values dropped from {dataset_name}!")

        if st.checkbox(f"Encode Categorical Variables in {dataset_name}"):
            label_encoders = {}
            for col in data.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])#label encoding
                label_encoders[col] = le
            st.success(f"Categorical variables encoded in {dataset_name}!")

        if st.checkbox(f"Scale Features in {dataset_name}"):
            scaler = StandardScaler()
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns #standardization
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            st.success(f"Features scaled in {dataset_name}!")
        return data

    # Process both datasets
    col1, col2 = st.columns(2)
    with col1:
        data1 = preprocess_data(data1, "Dataset 1")
    with col2:
        data2 = preprocess_data(data2, "Dataset 2")

    # Visualizations Section
    def visualize_data(data, dataset_name):
        st.subheader(f"Visualizations - {dataset_name}")

        if st.checkbox(f"Show Correlation Heatmap for {dataset_name}"):
            plt.figure(figsize=(10, 8))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)

        if st.checkbox(f"Show Pairplot for {dataset_name}"):
            st.write("Generating pairplot. This may take some time...")
            sns.pairplot(data)
            st.pyplot(plt)

        if st.checkbox(f"Show Box Plot for {dataset_name}"):
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            column = st.selectbox(f"Select Column for Box Plot in {dataset_name}", numeric_cols)
            plt.figure(figsize=(8, 6))
            sns.boxplot(data[column])
            st.pyplot(plt)

        if st.checkbox(f"Show Count Plot for {dataset_name}"):
            categorical_cols = data.select_dtypes(include=['object']).columns
            column = st.selectbox(f"Select Column for Count Plot in {dataset_name}", categorical_cols)
            plt.figure(figsize=(8, 6))
            sns.countplot(data[column])
            st.pyplot(plt)

        if st.checkbox(f"Show Value Distribution Table for {dataset_name}"):
            st.write("**Value Distribution:**")
            st.write(data.describe(include='all'))

    col1, col2 = st.columns(2)
    with col1:
        visualize_data(data1, "Dataset 1")
    with col2:
        visualize_data(data2, "Dataset 2")

    # Train-Test Split and Model Training
    def train_model(data, dataset_name):
        st.subheader(f"Model Training - {dataset_name}")

        # Use a unique key for the target column selector
        target_col = st.selectbox(
            f"Select Target Column for {dataset_name}",
            data.columns,
            key=f"{dataset_name}_target_col",
        )

        # Check target type and warn if it's continuous
        if data[target_col].dtype in ["float64", "int64"] and len(data[target_col].unique()) > 10:
            st.warning(f"The target column '{target_col}' seems to be continuous. Converting to discrete classes.")
            threshold = st.slider(f"Threshold for Binarization of {target_col} in {dataset_name}", 
                                float(data[target_col].min()), 
                                float(data[target_col].max()), 
                                value=float(data[target_col].mean()))
            data[target_col] = (data[target_col] >= threshold).astype(int)
            st.success(f"Converted '{target_col}' to binary classes using threshold {threshold:.2f}.")

        # Use a unique key for the test size slider
        test_size = st.slider(
            f"Test Size for {dataset_name} (%)", 10, 50, 20, key=f"{dataset_name}_test_size"
        ) / 100

        # Use a unique key for the random state input
        random_state = st.number_input(
            f"Random State for {dataset_name}", value=42, key=f"{dataset_name}_random_state"
        )

        X = data.drop(columns=[target_col])
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        st.success(f"Data split for {dataset_name}!")

        lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_train)

        st.write("**Classification Report:**")
        st.text(classification_report(y_train, y_pred))

        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_train, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(plt)

        roc_value = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
        st.write(f"**ROC-AUC Score:** {roc_value:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        train_model(data1, "Dataset 1")
    with col2:
        train_model(data2, "Dataset 2")

else:
    st.warning("Please upload two CSV files to proceed.")
