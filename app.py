import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# Main function
def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms poisonous or edible? 🍄")
    st.sidebar.markdown("Are your mushrooms poisonous or edible? 🍄")

    # Load data
    @st.cache_data
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    # Split data
    @st.cache_data
    def split(df):
        y = df.type
        x = df.drop(columns=["type"])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    # Plot metrics
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, model.predict(x_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots()  # Create a figure and axis
            disp.plot(ax=ax, cmap=plt.cm.Blues)  # Plot on the axis
            st.pyplot(fig)  # Pass the figure explicitly

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()  # Create a figure and axis
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)  # Plot on the axis
            st.pyplot(fig)  # Pass the figure explicitly

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()  # Create a figure and axis
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)  # Plot on the axis
            st.pyplot(fig)  # Pass the figure explicitly

    # Load and split data
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ["edible", "poisonous"]

    # Sidebar options
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("SVM", "Logistic Regression", "Random Forest"))

    # SVM Classifier
    if classifier == "SVM":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key="C")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")

        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        )

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("SVM Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(float(accuracy), 2))
            st.write("Precision:", round(float(precision_score(y_test, y_pred)), 2))
            st.write("Recall:", round(float(recall_score(y_test, y_pred)), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    # Logistic Regression Classifier
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key="C_LR")
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")

        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        )

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(float(accuracy), 2))
            st.write("Precision:", round(float(precision_score(y_test, y_pred)), 2))
            st.write("Recall:", round(float(recall_score(y_test, y_pred)), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    # Random Forest Classifier
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees", 10, 500, step=10, key="n_estimators")
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples", (True, False), key="bootstrap")

        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        )

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1
            )
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(float(accuracy), 2))
            st.write("Precision:", round(float(precision_score(y_test, y_pred)), 2))
            st.write("Recall:", round(float(recall_score(y_test, y_pred)), 2))

            plot_metrics(metrics, model, x_test, y_test, class_names)

    # Show raw data
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(df)


if __name__ == "__main__":
    main()