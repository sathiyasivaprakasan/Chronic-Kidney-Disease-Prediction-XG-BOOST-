import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb  # Import XGBoost

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Streamlit app title
st.title("Chronic Kidney Disease Prediction with Model Comparison")

# File uploader for dataset upload
data_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if data_file is not None:
    # Read the uploaded dataset
    df = pd.read_csv(data_file)
    st.write("### Uploaded Dataset:")
    st.dataframe(df.head())

    # Preprocessing steps
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)  # Drop ID column if exists

    # Rename columns (ensure consistency)
    df.columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                  'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']

    # Convert columns to numeric where needed
    df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
    df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
    df['rc'] = pd.to_numeric(df['rc'], errors='coerce')

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Split dataset
    X = df.drop('classification', axis=1)
    y = df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X.columns[selector.get_support()]

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_test_selected = scaler.transform(X_test_selected)

    # Train models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_selected, y_train)
    xgb_model = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_selected, y_train)

    # Predictions
    rf_predictions = rf_model.predict(X_test_selected)
    xgb_predictions = xgb_model.predict(X_test_selected)

    # Model evaluation
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)

    # Display accuracy comparison
    st.write("### Model Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(["Random Forest", "XGBoost"], [rf_accuracy, xgb_accuracy], color=['blue', 'red'])
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)

    # Confusion Matrices
    st.write("### Confusion Matrices")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Random Forest Confusion Matrix")
    sns.heatmap(confusion_matrix(y_test, xgb_predictions), annot=True, fmt='d', cmap='Reds', ax=axes[1])
    axes[1].set_title("XGBoost Confusion Matrix")
    st.pyplot(fig)

    # Model selection for prediction
    model_choice = st.selectbox("Select Model for Prediction", ["Random Forest", "XGBoost"])

    # Function to handle predictions
    def onPredict(model, data):
        """
        Function to make predictions using the selected model.
        Args:
            model: The trained model (Random Forest or XGBoost).
            data: The input data for prediction.
        Returns:
            List of predictions.
        """
        try:
            predictions = model.predict(data)
            return predictions
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return []

    # Individual Record Entry
    st.write("### Predict CKD for an Individual Record")
    individual_data = {}
    for col in X.columns:
        if col in df.select_dtypes(include=['object']).columns:
            unique_values = df[col].unique()
            individual_data[col] = st.selectbox(f"{col}", options=unique_values)
        else:
            individual_data[col] = st.number_input(f"{col}", value=float(df[col].median()))

    if st.button("Predict CKD"):
        individual_df = pd.DataFrame([individual_data])
        for col in individual_df.select_dtypes(include='object').columns:
            individual_df[col] = label_encoder.transform(individual_df[col])
        individual_df = individual_df[selected_features]
        individual_df = scaler.transform(individual_df)

        # Use the onPredict function
        if model_choice == "Random Forest":
            prediction = onPredict(rf_model, individual_df)
        else:
            prediction = onPredict(xgb_model, individual_df)

        if prediction:
            st.write("### Individual Prediction Result")
            st.write("CKD" if prediction[0] == 1 else "No CKD")

    # Predict CKD for all records
    st.write("### Predict CKD for All Records")
    if st.button("Predict for All Records"):
        # Use the onPredict function
        if model_choice == "Random Forest":
            all_predictions = onPredict(rf_model, X_test_selected)
        else:
            all_predictions = onPredict(xgb_model, X_test_selected)

        if all_predictions:
            X_test_with_predictions = X_test[selected_features].copy()
            X_test_with_predictions['Prediction'] = ["CKD" if pred == 1 else "No CKD" for pred in all_predictions]
            st.write(X_test_with_predictions)
else:
    st.write("Please upload a dataset to proceed.")