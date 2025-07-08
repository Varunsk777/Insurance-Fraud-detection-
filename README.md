# Fraud Detection Dashboard

This project provides an interactive dashboard for fraud detection using **Streamlit**. It allows users to upload a CSV file, preprocess the data, visualize it, and train a **Random Forest Classifier** to detect fraudulent claims. A companion **Jupyter Notebook** is also included to generate synthetic fraud data and train a model.

---

## 🚀 Features

- 📁 **Upload and View Dataset**: Upload a CSV file and preview its contents.
- 🧹 **Data Preprocessing**: Handle missing values, encode categorical variables, and scale features.
- 📊 **Visualizations**:
  - Correlation heatmaps
  - Pairplots
  - Target (Fraudulent) distribution plots
- 🤖 **Model Training**:
  - Train-test split
  - Random Forest Classifier training
- 📈 **Model Evaluation**:
  - Classification report
  - Confusion matrix
  - ROC-AUC score

---

## 📦 Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
## 🖥️ Usage
- Running the Dashboard
To launch the Streamlit dashboard, execute:
```bash
streamlit run dashboard.py
```
Then, open your web browser and navigate to http://localhost:8501 to interact with the dashboard
## 🧾 Data Schema
- The dashboard accepts any CSV file with appropriate columns for fraud detection.

## Example Columns from Synthetic Data:
- Customer ID: Unique identifier for each customer

- Age: Age of the policyholder

- Annual Income: Policyholder’s annual income

- Region: Geographical region

- Policy ID, Policy Term, Policy Age

- Premium Amount, Coverage Amount

- Previous Claims Count

- Claim ID, Claim Date, Claim Amount, Report Delay

- Claim Status: Pending / Approved / Rejected

- High-Value Claim, Multiple Claims Indicator, Suspicious Region, Inconsistent Information, Third-Party Involvement

- Fraudulent: Target variable (0 = Non-Fraud, 1 = Fraud)

## Engineered Features:
- Claim to Premium Ratio

- Income to Coverage Ratio

- Policy Age Group

- High Risk Region

For detailed column descriptions, refer to the notebook.

## 🧪 Dependencies
Listed in requirements.txt:

- streamlit
- pandas
- numpy
- faker
- scipy
- seaborn
- matplotlib
- scikit-learn


